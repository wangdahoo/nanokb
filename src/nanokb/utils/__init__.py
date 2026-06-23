"""utils 工具层：hashing / io（原子写 + staging 切换）/ tokenize。

为流水线提供三项基础设施：
- 增量检测指纹：``sha256_file``
- 原子事务：``atomic_write_text`` / ``atomic_write_json`` / ``staging_swap``
- 精确 token 计数：``count_tokens``（tiktoken）
"""

from __future__ import annotations

from nanokb.utils.hashing import sha256_file
from nanokb.utils.io import (
    STAGING_FILES,
    atomic_write_json,
    atomic_write_text,
    staging_swap,
)
from nanokb.utils.tokenize import count_tokens

__all__ = [
    "STAGING_FILES",
    "atomic_write_json",
    "atomic_write_text",
    "count_tokens",
    "sha256_file",
    "staging_swap",
]
