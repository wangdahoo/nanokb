"""文档加载器单测（方案 §3.4.2 + §3.6 + s1-feat-003 全部 AC）。

覆盖：
- AC #1：.md/.txt 直接 read_text 返回纯文本
- AC #2：.pdf 经 unstructured.partition 抽取（用 monkeypatch 打桩，不触网）
- AC #3：注册多个 loader 的 LoaderRegistry，load 返回首个 supports 的 loader 结果
- AC #4：不支持的扩展名（.xyz）抛 UnsupportedFormatError
"""

from __future__ import annotations

from pathlib import Path

import pytest

import nanokb.loaders as loaders_pkg
from nanokb.loaders import (
    DocumentLoader,
    LoaderRegistry,
    UnstructuredLoader,
    UnsupportedFormatError,
)

FIXTURES_RAW = Path(__file__).resolve().parent.parent / "fixtures" / "raw"


# --------------------------------------------------------------------------- #
# 测试辅助：伪 loader（实现 DocumentLoader Protocol）与伪 unstructured 元素
# --------------------------------------------------------------------------- #


class _FakeCodeLoader:
    """模拟 s1-feat-010 的 CodeLoader：声明支持 .py，返回标记文本。

    用于验证 LoaderRegistry 多 loader 分发与"首个 supports 胜出"语义。
    """

    def __init__(self) -> None:
        self.loaded: list[Path] = []

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".py"

    def load(self, path: Path) -> str:
        self.loaded.append(path)
        return f"<code:{path.name}>"


class _AlwaysLoader:
    """声明支持任意路径的兜底 loader（注册顺序靠后用于验证优先级）。"""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def supports(self, path: Path) -> bool:
        return True

    def load(self, path: Path) -> str:  # noqa: ARG002
        return f"<{self.tag}>"


class _FakeElement:
    """模拟 unstructured 的 TextBaseElement：仅需 .text 属性。"""

    def __init__(self, text: str) -> None:
        self.text = text


def _make_fake_partition(captured: dict[str, object], elements: list[_FakeElement]) -> object:
    """构造一个记录 filename 入参的伪 partition 函数。"""

    def _fake_partition(filename: str | None = None, **kwargs: object) -> list[_FakeElement]:
        captured["filename"] = filename
        captured["kwargs"] = kwargs
        return elements

    return _fake_partition


# --------------------------------------------------------------------------- #
# UnstructuredLoader.supports
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("ext", [".md", ".txt", ".pdf", ".docx", ".MD", ".PDF"])
def test_unstructured_loader_supports_known_extensions(ext: str) -> None:
    loader = UnstructuredLoader()
    assert loader.supports(Path(f"doc{ext}"))


@pytest.mark.parametrize("ext", [".xyz", ".py", ".json", ""])
def test_unstructured_loader_does_not_support_unknown_extensions(ext: str) -> None:
    loader = UnstructuredLoader()
    assert not loader.supports(Path(f"doc{ext}"))


# --------------------------------------------------------------------------- #
# AC #1：.md/.txt 直接读
# --------------------------------------------------------------------------- #


def test_unstructured_loader_loads_md_returns_plain_text() -> None:
    """AC #1：.md 文件返回纯文本内容（与 read_text 一致）。"""
    loader = UnstructuredLoader()
    sample = FIXTURES_RAW / "sample.md"
    assert sample.exists(), f"fixture missing: {sample}"

    text = loader.load(sample)
    assert text == sample.read_text(encoding="utf-8")
    assert "Transformer" in text
    assert "Self-Attention" in text


def test_unstructured_loader_loads_txt_returns_plain_text() -> None:
    """AC #1：.txt 文件返回纯文本内容。"""
    loader = UnstructuredLoader()
    sample = FIXTURES_RAW / "sample.txt"
    assert sample.exists(), f"fixture missing: {sample}"

    text = loader.load(sample)
    assert text == sample.read_text(encoding="utf-8")
    assert "Attention Is All You Need" in text


def test_unstructured_loader_loads_md_from_tmp(tmp_path: Path) -> None:
    """AC #1：对临时 .md 文件加载，内容逐字一致（含中英文混合）。"""
    loader = UnstructuredLoader()
    f = tmp_path / "note.md"
    payload = "# 标题\n\n中英文 mixed content here.\n"
    f.write_text(payload, encoding="utf-8")

    assert loader.load(f) == payload


def test_unstructured_loader_loads_txt_preserves_newlines(tmp_path: Path) -> None:
    """AC #1：换行与空白逐字保留，不经过任何 partition。"""
    loader = UnstructuredLoader()
    f = tmp_path / "a.txt"
    payload = "line1\nline2\n\nline4"
    f.write_text(payload, encoding="utf-8")

    assert loader.load(f) == payload


# --------------------------------------------------------------------------- #
# AC #2：.pdf/.docx 经 unstructured.partition（monkeypatch 打桩）
# --------------------------------------------------------------------------- #


def test_unstructured_loader_loads_pdf_via_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC #2：.pdf 经 unstructured.partition.auto.partition 抽取，元素 text 以空行连接。"""
    import unstructured.partition.auto as ua

    captured: dict[str, object] = {}
    fake = _make_fake_partition(
        captured,
        [_FakeElement("Para one"), _FakeElement("Para two"), _FakeElement("第三段")],
    )
    monkeypatch.setattr(ua, "partition", fake)

    # 不需要真实 pdf：partition 已被打桩，文件内容不会被解析
    fake_pdf = tmp_path / "doc.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 not a real pdf")

    loader = UnstructuredLoader()
    text = loader.load(fake_pdf)

    assert text == "Para one\n\nPara two\n\n第三段"
    # 验证确实调用了 partition，并以 filename= 形式传入路径
    assert captured["filename"] == str(fake_pdf)


def test_unstructured_loader_loads_docx_via_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC #2：.docx 同样走 unstructured.partition 通道。"""
    import unstructured.partition.auto as ua

    captured: dict[str, object] = {}
    fake = _make_fake_partition(captured, [_FakeElement("Hello docx")])
    monkeypatch.setattr(ua, "partition", fake)

    fake_docx = tmp_path / "doc.docx"
    fake_docx.write_bytes(b"PK\x03\x04 not a real docx")

    loader = UnstructuredLoader()
    assert loader.load(fake_docx) == "Hello docx"
    assert captured["filename"] == str(fake_docx)


def test_unstructured_loader_partition_skips_empty_text_elements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """空/None text 元素被跳过，不产生多余空段。"""
    import unstructured.partition.auto as ua

    monkeypatch.setattr(
        ua,
        "partition",
        _make_fake_partition(
            {},
            [_FakeElement("keep"), _FakeElement(""), _FakeElement("also keep")],
        ),
    )
    loader = UnstructuredLoader()
    fake_pdf = tmp_path / "x.pdf"
    fake_pdf.write_bytes(b"pdf")

    assert loader.load(fake_pdf) == "keep\n\nalso keep"


def test_unstructured_loader_does_not_call_partition_for_text_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """.md/.txt 直读，绝不经 unstructured.partition（保证离线/零开销）。"""
    import unstructured.partition.auto as ua

    calls: list[str] = []

    def _should_not_be_called(**_kwargs: object) -> list[_FakeElement]:
        calls.append("called")
        return []

    monkeypatch.setattr(ua, "partition", _should_not_be_called)

    loader = UnstructuredLoader()
    f = tmp_path / "a.md"
    f.write_text("plain markdown", encoding="utf-8")

    assert loader.load(f) == "plain markdown"
    assert calls == []


def test_unstructured_loader_load_unsupported_extension_raises(tmp_path: Path) -> None:
    """直接对不支持的扩展名调 load 抛 UnsupportedFormatError（与 registry 一致）。"""
    loader = UnstructuredLoader()
    f = tmp_path / "a.xyz"
    f.write_text("x", encoding="utf-8")

    with pytest.raises(UnsupportedFormatError):
        loader.load(f)


# --------------------------------------------------------------------------- #
# AC #3：LoaderRegistry 多 loader 分发，首个 supports 胜出
# --------------------------------------------------------------------------- #


def test_registry_load_returns_first_supporting_loader(tmp_path: Path) -> None:
    """AC #3：注册多 loader，registry.load 返回首个 supports 的 loader 结果。"""
    reg = LoaderRegistry()
    code_loader = _FakeCodeLoader()
    unstructured = UnstructuredLoader()
    fallback = _AlwaysLoader("fallback")
    reg.register(unstructured)
    reg.register(code_loader)
    reg.register(fallback)

    md = tmp_path / "a.md"
    md.write_text("MD CONTENT", encoding="utf-8")
    py = tmp_path / "b.py"
    py.write_text("print('hi')", encoding="utf-8")
    rst = tmp_path / "c.rst"
    rst.write_text("RST", encoding="utf-8")

    # .md → 首个 supports 的是 UnstructuredLoader
    assert reg.load(md) == "MD CONTENT"
    assert code_loader.loaded == []
    # .py → UnstructuredLoader 不支持，落到 _FakeCodeLoader
    assert reg.load(py) == "<code:b.py>"
    # .rst → 仅 _AlwaysLoader 支持
    assert reg.load(rst) == "<fallback>"


def test_registry_load_first_supporting_wins_over_later_supporting(tmp_path: Path) -> None:
    """AC #3：当多个 loader 都 supports 同一路径时，注册顺序靠前者胜出。"""
    reg = LoaderRegistry()
    reg.register(_AlwaysLoader("first"))
    reg.register(_AlwaysLoader("second"))

    assert reg.load(tmp_path / "any.md") == "<first>"


def test_registry_load_uses_unstructured_for_md_txt_fixture() -> None:
    """AC #3 集成：注册 UnstructuredLoader 后能加载真实 fixture .md。"""
    reg = LoaderRegistry()
    reg.register(UnstructuredLoader())

    sample = FIXTURES_RAW / "sample.md"
    assert sample.exists()
    text = reg.load(sample)
    assert "Transformer" in text
    assert text == sample.read_text(encoding="utf-8")


def test_registry_register_preserves_order_and_is_inspectable() -> None:
    """register 按顺序追加，loaders 属性提供只读视图。"""
    reg = LoaderRegistry()
    assert reg.loaders == ()

    a = _AlwaysLoader("a")
    b = _AlwaysLoader("b")
    reg.register(a)
    reg.register(b)

    assert reg.loaders == (a, b)


# --------------------------------------------------------------------------- #
# AC #4：不支持的扩展名 → UnsupportedFormatError
# --------------------------------------------------------------------------- #


def test_registry_load_unsupported_extension_raises(tmp_path: Path) -> None:
    """AC #4：.xyz 无 loader 支持，registry.load 抛 UnsupportedFormatError。"""
    reg = LoaderRegistry()
    reg.register(UnstructuredLoader())

    f = tmp_path / "weird.xyz"
    f.write_text("nope", encoding="utf-8")

    with pytest.raises(UnsupportedFormatError):
        reg.load(f)


def test_registry_load_empty_registry_raises(tmp_path: Path) -> None:
    """空 registry 对任意路径均抛 UnsupportedFormatError。"""
    reg = LoaderRegistry()
    with pytest.raises(UnsupportedFormatError):
        reg.load(tmp_path / "anything.md")


def test_registry_load_py_unsupported_without_code_loader(tmp_path: Path) -> None:
    """仅注册 UnstructuredLoader 时 .py 不被支持（CodeLoader 在 s1-feat-010 接入）。"""
    reg = LoaderRegistry()
    reg.register(UnstructuredLoader())

    f = tmp_path / "a.py"
    f.write_text("x = 1", encoding="utf-8")
    with pytest.raises(UnsupportedFormatError):
        reg.load(f)


# --------------------------------------------------------------------------- #
# 包导出与 Protocol 可检性
# --------------------------------------------------------------------------- #


def test_package_exports_public_api() -> None:
    """__init__ 导出四个公共符号，便于上层 from nanokb.loaders import ...。"""
    for name in ("DocumentLoader", "LoaderRegistry", "UnstructuredLoader", "UnsupportedFormatError"):
        assert hasattr(loaders_pkg, name)


def test_unstructured_loader_satisfies_protocol() -> None:
    """UnstructuredLoader 结构性满足 DocumentLoader Protocol（@runtime_checkable）。"""
    loader = UnstructuredLoader()
    assert isinstance(loader, DocumentLoader)


def test_fake_code_loader_satisfies_protocol() -> None:
    """自定义 loader 结构性满足 DocumentLoader Protocol（扩展点有效）。"""
    assert isinstance(_FakeCodeLoader(), DocumentLoader)
