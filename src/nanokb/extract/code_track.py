"""代码轨 tree-sitter 抽取（方案 §3.4.3 + §4 阶段 4，Feature s1-feat-010）。

``CodeTrack`` 用 tree-sitter（经 ``tree-sitter-languages`` 预编译 grammar 包，Medium #6）
对源码做**确定性**结构抽取，产出三类关系边：

- ``defines``：(文件模块, defines, 顶层函数/类)。
- ``contains``：(类, contains, 方法)（head==tail 自环跳过）。
- ``calls``：(函数/方法, calls, 被调用名)。

节点描述由签名派生：

- 函数：``function foo(a, b)``（取 ``parameters`` 字段文本）。
- 类：``class Foo``。
- 方法：``method Foo.bar(self, x)``（带所属类上下文）。
- 仅被调用（无定义）：``function bar``（仅名）。

约束：

- **零 Token**：``extract`` 内不调用 ``llm``（``llm`` 仅在 ``SemanticTrack`` 构造期注入）。
  ``confidence`` 固定 ``EXTRACTED``，``track`` 固定 ``Track.CODE``，``chunk_index`` 为 ``None``。
- 限定 Python / JavaScript / Java 三语言；可用语言受 ``settings.code_languages`` 过滤。
- ``CodeTrack(settings)`` 仅依赖 settings，满足 ``Extractor`` Protocol。
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

from tree_sitter import Node, Parser

# tree-sitter-languages 不随包发布 py.typed（Medium #6 预编译 grammar 包）。
from tree_sitter_languages import get_parser  # type: ignore[import-untyped]

from nanokb.config import Settings
from nanokb.models import (
    Concept,
    Confidence,
    Document,
    ExtractionResult,
    Track,
    Triple,
)

logger = logging.getLogger("nanokb")

# ── 关系名常量 ────────────────────────────────────────────────────────

REL_DEFINES = "defines"
REL_CONTAINS = "contains"
REL_CALLS = "calls"

#: 三种关系的固定节点类型描述前缀
_KIND_FUNCTION = "function"
_KIND_CLASS = "class"
_KIND_METHOD = "method"
_KIND_MODULE = "module"


# ── 语言配置 ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LangSpec:
    """单语言的 tree-sitter 抽取配置。

    Attributes:
        language: ``tree_sitter_languages.get_language`` 接受的语言名。
        suffix: 对应文件扩展名（含点，小写）。
        function_defs: 视为「函数」定义的 tree-sitter 节点类型集合。
        class_defs: 视为「类」定义的节点类型集合。
        method_defs: 视为「方法」（类成员）定义的节点类型集合；Python 用空集——
            其方法节点类型与函数相同（``function_definition``），靠 enclosing class 上下文判定。
        call_types: 视为「调用」的节点类型集合。
        contextual_methods: True 表示函数节点在类作用域内应降级为方法（Python）。
        extract_callee: 从调用节点抽取被调用名；无法识别返回 None。
    """

    language: str
    suffix: str
    function_defs: frozenset[str]
    class_defs: frozenset[str]
    method_defs: frozenset[str]
    call_types: frozenset[str]
    contextual_methods: bool
    extract_callee: Callable[[Node], str | None]
    enabled_attr: str = field(default="")  # settings.code_languages 中对应的启用键名


def _python_callee(node: Node) -> str | None:
    """Python ``call`` 节点 → 被调用名。

    ``function`` 字段为 ``identifier`` 时直接取名；为 ``attribute``（obj.method）时取末段。
    """
    func = node.child_by_field_name("function")
    if func is None:
        return None
    if func.type == "identifier":
        return _decode(func.text)
    if func.type == "attribute":
        # attribute 节点文本形如 "obj.method"，取末段保留方法名
        return _decode(func.text).rsplit(".", 1)[-1]
    return None


def _javascript_callee(node: Node) -> str | None:
    """JavaScript ``call_expression`` → 被调用名。

    ``function`` 字段为 ``identifier`` 时直取名；为 ``member_expression``（obj.m）时取
    ``property`` 字段文本。
    """
    func = node.child_by_field_name("function")
    if func is None:
        return None
    if func.type == "identifier":
        return _decode(func.text)
    if func.type == "member_expression":
        prop = func.child_by_field_name("property")
        if prop is not None:
            return _decode(prop.text)
        return _decode(func.text).rsplit(".", 1)[-1]
    return None


def _java_callee(node: Node) -> str | None:
    """Java ``method_invocation`` → 被调用名（``name`` 字段，如 helper.run 取 run）。"""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return None
    return _decode(name_node.text)


_PYTHON_SPEC = LangSpec(
    language="python",
    suffix=".py",
    function_defs=frozenset({"function_definition"}),
    class_defs=frozenset({"class_definition"}),
    method_defs=frozenset(),
    call_types=frozenset({"call"}),
    contextual_methods=True,
    extract_callee=_python_callee,
    enabled_attr="python",
)

_JAVASCRIPT_SPEC = LangSpec(
    language="javascript",
    suffix=".js",
    function_defs=frozenset({"function_declaration"}),
    class_defs=frozenset({"class_declaration"}),
    method_defs=frozenset({"method_definition"}),
    call_types=frozenset({"call_expression"}),
    contextual_methods=False,
    extract_callee=_javascript_callee,
    enabled_attr="javascript",
)

_JAVA_SPEC = LangSpec(
    language="java",
    suffix=".java",
    function_defs=frozenset(),
    class_defs=frozenset({"class_declaration"}),
    method_defs=frozenset({"method_declaration"}),
    call_types=frozenset({"method_invocation"}),
    contextual_methods=False,
    extract_callee=_java_callee,
    enabled_attr="java",
)

#: 全部受支持语言（按扩展名索引）
_ALL_SPECS: tuple[LangSpec, ...] = (_PYTHON_SPEC, _JAVASCRIPT_SPEC, _JAVA_SPEC)


def supported_code_suffixes() -> frozenset[str]:
    """返回 CodeTrack 受支持的全部代码扩展名（供 pipeline 分发用）。"""
    return frozenset(spec.suffix for spec in _ALL_SPECS)


def spec_for_suffix(suffix: str) -> LangSpec | None:
    """按扩展名返回 ``LangSpec``；未受支持返回 ``None``。"""
    norm = suffix.lower()
    for spec in _ALL_SPECS:
        if spec.suffix == norm:
            return spec
    return None


# ── 作用域栈帧 ────────────────────────────────────────────────────────


@dataclass
class _Scope:
    """词法作用域栈帧：当前所在的函数/方法/类名 + 种类。"""

    name: str
    kind: str  # 'function' | 'method' | 'class'


# ── 工具：字节文本解码 ────────────────────────────────────────────────


def _decode(raw: bytes | None) -> str:
    """tree-sitter ``Node.text`` 为 bytes；utf-8 解码并以空格折叠空白。"""
    if raw is None:
        return ""
    try:
        return raw.decode("utf-8", errors="replace")
    except Exception:  # pragma: no cover - decode with replace 极少抛
        return ""


def _collapse_ws(text: str) -> str:
    """折叠连续空白为单空格并去前后空白（用于 parameters 文本规范化）。"""
    return " ".join(text.split())


# ── CodeTrack ─────────────────────────────────────────────────────────


class CodeTrack:
    """代码轨抽取器：tree-sitter 确定性结构抽取（零 Token）。

    构造仅注入 ``settings``（不注入 ``llm``，方案 §3.4.3 注：``llm`` 仅在
    ``SemanticTrack`` 构造期注入）。``settings.code_languages`` 过滤可用语言；未启用的
    语言对应文件返回空结果。

    Parser 按 language 缓存在实例上（同实例多文件复用）。watch 单 worker 线程消费模型下，
    实例不跨线程共享 → 无并发风险。
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        enabled = {lang.lower() for lang in settings.code_languages}
        # 仅保留 settings.code_languages 启用的语言
        self._specs: tuple[LangSpec, ...] = tuple(
            spec for spec in _ALL_SPECS if spec.enabled_attr in enabled
        )
        self._parsers: dict[str, Parser] = {}

    def extract(self, doc: Document) -> ExtractionResult:
        """用 tree-sitter 解析 ``doc.content``，产出 defines/contains/calls 三元组。

        ``doc`` 的扩展名不在受支持集合（或对应语言被 settings 过滤）时返回空结果。
        """
        spec = spec_for_suffix(doc.path.suffix)
        if spec is None or spec not in self._specs:
            return ExtractionResult()

        source_file = str(doc.path)
        parser = self._get_parser(spec.language)
        try:
            tree = parser.parse(doc.content.encode("utf-8", errors="replace"))
        except Exception:
            logger.exception(
                "code_track: tree-sitter parse failed for %s (%s)",
                source_file,
                spec.language,
                extra={"stage": "code-track", "file": source_file},
            )
            return ExtractionResult()

        triples: list[Triple] = []
        concepts: dict[str, Concept] = {}
        file_node = doc.path.stem or doc.path.name

        # 文件模块节点（作为 defines 关系的 head）
        concepts[file_node] = Concept(
            name=file_node,
            description=f"{_KIND_MODULE} {file_node} ({spec.suffix.lstrip('.')})",
            source_file=source_file,
            node_type=_KIND_MODULE,
            confidence=Confidence.EXTRACTED,
        )

        scope_stack: list[_Scope] = []
        self._walk(
            tree.root_node,
            spec,
            scope_stack,
            file_node,
            source_file,
            triples,
            concepts,
        )

        return ExtractionResult(triples=triples, concepts=list(concepts.values()))

    # ── 遍历 ──────────────────────────────────────────────────────────

    def _walk(
        self,
        node: Node,
        spec: LangSpec,
        scope_stack: list[_Scope],
        file_node: str,
        source_file: str,
        triples: list[Triple],
        concepts: dict[str, Concept],
    ) -> None:
        """深度优先遍历：识别定义/调用节点并记录关系，维护作用域栈。

        对定义节点：记录 concept + defines/contains 三元组 → 压栈 → 递归 → 出栈。
        对调用节点：记录 calls 三元组，仍递归子节点（捕获嵌套调用 ``f(g(x))``）。
        """
        pushed = False
        kind = self._classify_def(node, spec, scope_stack)

        if kind is not None:
            name, params = self._def_name_and_params(node)
            if name:
                self._record_definition(
                    name=name,
                    kind=kind,
                    params=params,
                    scope_stack=scope_stack,
                    file_node=file_node,
                    source_file=source_file,
                    triples=triples,
                    concepts=concepts,
                )
                scope_stack.append(_Scope(name=name, kind=kind))
                pushed = True
        elif node.type in spec.call_types:
            callee = spec.extract_callee(node)
            if callee:
                caller = self._caller(scope_stack, file_node)
                self._emit(triples, caller, REL_CALLS, callee, source_file)
                self._ensure_callee_concept(callee, source_file, concepts)

        for child in node.children:
            self._walk(child, spec, scope_stack, file_node, source_file, triples, concepts)

        if pushed:
            scope_stack.pop()

    # ── 定义节点处理 ──────────────────────────────────────────────────

    def _classify_def(self, node: Node, spec: LangSpec, scope_stack: list[_Scope]) -> str | None:
        """判定节点是否为定义，返回 'class' / 'method' / 'function'，否则 None。

        Python 的 ``function_definition`` 在类作用域内降级为 method（contextual_methods）。
        """
        ntype = node.type
        if ntype in spec.class_defs:
            return _KIND_CLASS
        if ntype in spec.method_defs:
            return _KIND_METHOD
        if ntype in spec.function_defs:
            if spec.contextual_methods and self._enclosing_class(scope_stack) is not None:
                return _KIND_METHOD
            return _KIND_FUNCTION
        return None

    @staticmethod
    def _def_name_and_params(node: Node) -> tuple[str, str]:
        """从定义节点抽取 (name, parameters_text)；parameters 缺失为空串。

        ``parameters`` 字段文本形如 ``(a, b)``（含外层括号），此处剥离括号并折叠空白，
        使描述模板 ``{name}({params})`` 不产生双重括号。
        """
        name_node = node.child_by_field_name("name")
        name = _decode(name_node.text) if name_node is not None else ""
        params_node = node.child_by_field_name("parameters")
        if params_node is None:
            return name, ""
        params = _collapse_ws(_decode(params_node.text))
        # 剥离 parameters 节点的外层括号
        if len(params) >= 2 and params[0] == "(" and params[-1] == ")":
            params = params[1:-1].strip()
        return name, params

    def _record_definition(
        self,
        *,
        name: str,
        kind: str,
        params: str,
        scope_stack: list[_Scope],
        file_node: str,
        source_file: str,
        triples: list[Triple],
        concepts: dict[str, Concept],
    ) -> None:
        """记录定义节点对应的 concept 与 defines/contains 三元组。"""
        enclosing_class = self._enclosing_class(scope_stack)

        if kind == _KIND_CLASS:
            description = f"{_KIND_CLASS} {name}"
            node_type = _KIND_CLASS
        elif kind == _KIND_METHOD:
            if enclosing_class:
                description = f"{_KIND_METHOD} {enclosing_class}.{name}({params})"
                self._emit(triples, enclosing_class, REL_CONTAINS, name, source_file)
            else:
                description = f"{_KIND_METHOD} {name}({params})"
            node_type = _KIND_METHOD
        else:  # function
            description = f"{_KIND_FUNCTION} {name}({params})"
            node_type = _KIND_FUNCTION

        # 顶层函数/类 → file defines entity
        if not scope_stack and kind in (_KIND_FUNCTION, _KIND_CLASS):
            self._emit(triples, file_node, REL_DEFINES, name, source_file)

        concepts[name] = Concept(
            name=name,
            description=description,
            source_file=source_file,
            node_type=node_type,
            confidence=Confidence.EXTRACTED,
        )

    def _ensure_callee_concept(
        self, callee: str, source_file: str, concepts: dict[str, Concept]
    ) -> None:
        """为仅被调用（无定义）的实体补一个名派生 concept；已有定义概念则保留。"""
        if callee in concepts:
            return
        concepts[callee] = Concept(
            name=callee,
            description=f"{_KIND_FUNCTION} {callee}",
            source_file=source_file,
            node_type=_KIND_FUNCTION,
            confidence=Confidence.EXTRACTED,
        )

    # ── 作用域辅助 ────────────────────────────────────────────────────

    @staticmethod
    def _enclosing_class(scope_stack: list[_Scope]) -> str | None:
        """返回最近的 class 作用域名；无则 None。"""
        for scope in reversed(scope_stack):
            if scope.kind == _KIND_CLASS:
                return scope.name
        return None

    @staticmethod
    def _caller(scope_stack: list[_Scope], file_node: str) -> str:
        """返回最近的函数/方法作用域名作为 caller；模块级调用回落到 file_node。"""
        for scope in reversed(scope_stack):
            if scope.kind in (_KIND_FUNCTION, _KIND_METHOD):
                return scope.name
        return file_node

    # ── Triple 工厂 / Parser 缓存 ─────────────────────────────────────

    @staticmethod
    def _emit(
        triples: list[Triple],
        head: str,
        relation: str,
        tail: str,
        source_file: str,
    ) -> None:
        """构造并追加一条三元组；head==tail 自环跳过。

        自环跳过的场景：(a) 类同名方法（Java constructor / 类名==方法名）；
        (b) Java 文件名与 public 类同名（file_node==class name）致 ``defines`` 自环；
        (c) 函数自调用 ``foo()`` 内调 ``foo()``。跳过这些保持图谱整洁且避免无意义边。
        """
        if head == tail:
            return
        triples.append(
            Triple(
                head=head,
                relation=relation,
                tail=tail,
                confidence=Confidence.EXTRACTED,
                source_file=source_file,
                track=Track.CODE,
                chunk_index=None,
            )
        )

    def _get_parser(self, language: str) -> Parser:
        """按 language 缓存 Parser；首次创建时屏蔽 tree-sitter 0.21 的 FutureWarning。"""
        cached = self._parsers.get(language)
        if cached is not None:
            return cached
        with warnings.catch_warnings():
            # tree-sitter 0.21.3 在 Language 构造路径上发出 deprecation FutureWarning
            # （与 tree-sitter-languages 1.10 的调用方式相关），不影响功能，屏蔽以保持日志洁净。
            warnings.simplefilter("ignore", FutureWarning)
            # tree_sitter_languages 无 py.typed → get_parser 返回 Any，cast 为 Parser。
            parser = cast(Parser, get_parser(language))
        self._parsers[language] = parser
        return parser


__all__ = [
    "CodeTrack",
    "LangSpec",
    "spec_for_suffix",
    "supported_code_suffixes",
]
