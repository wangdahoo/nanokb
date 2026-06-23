"""tiktoken 精确 token 计数（方案 §3.4.1 count_tokens + Opt #6）。

按 model 选择 tokenizer：
- OpenAI 已知 model：``tiktoken.encoding_for_model`` 精确匹配。
- 非 OpenAI model / 未知 model（如 glm-5.1）：fallback 到 ``cl100k_base``
  （通用 BPE，中英文覆盖较好，作为跨 provider 的统一兜底）。

为 chunker 分块与 prompt 上下文裁剪提供精确 token 计数，避免 chars/4 粗估导致的窗口溢出。

离线策略：cl100k_base 的 BPE 词表已内置在 ``utils/_tiktoken_cache/``（OpenAI 公共 CDN
在受限网络下不可达）。导入时将 ``TIKTOKEN_CACHE_DIR`` 指向该目录，tiktoken 直接命中
本地缓存而不再联网下载。若外部已设置该环境变量则予以尊重。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import tiktoken

# 默认模型：项目默认 LLM 为 glm-5.1（Zhipu）。tiktoken 不识别该名称，
# 由 _get_encoding fallback 到 cl100k_base，提供近似的精确 token 计数。
DEFAULT_MODEL = "glm-5.1"

_BUNDLED_CACHE = Path(__file__).resolve().parent / "_tiktoken_cache"
if _BUNDLED_CACHE.is_dir() and not os.environ.get("TIKTOKEN_CACHE_DIR"):
    os.environ["TIKTOKEN_CACHE_DIR"] = str(_BUNDLED_CACHE)


@lru_cache(maxsize=32)
def _get_encoding(model: str) -> tiktoken.Encoding:
    """获取 model 对应的 tiktoken encoding。

    OpenAI 已知 model 走 ``encoding_for_model``；KeyError 时 fallback 到 ``cl100k_base``。
    结果 LRU 缓存，避免重复构造（encoding 加载有 I/O 开销）。
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = DEFAULT_MODEL) -> int:
    """精确计数 ``text`` 在 ``model`` tokenizer 下的 token 数。

    使用真实 tiktoken BPE 编码长度（非 chars/4 粗估）。空字符串返回 0。
    """
    if not text:
        return 0
    return len(_get_encoding(model).encode(text))


__all__ = ["DEFAULT_MODEL", "count_tokens"]
