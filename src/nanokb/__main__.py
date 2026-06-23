"""Nano KB 模块入口，支持 `python -m nanokb`。"""

from __future__ import annotations

from nanokb.cli import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
