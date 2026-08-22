"""配置加载 —— 统一从 YAML 读取全部 Mock 参数（spec：代码不硬编码）。"""
from __future__ import annotations

import pathlib
from typing import Any

import yaml


def load_config(path: str | pathlib.Path | None = None) -> dict[str, Any]:
    """加载 YAML 配置；未指定时用项目默认 config/default.yaml。

    Args:
        path: YAML 文件路径；None 时回退到仓库根 config/default.yaml。

    Returns:
        解析后的 dict。
    """
    if path is None:
        path = pathlib.Path(__file__).resolve().parent.parent / "config" / "default.yaml"
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    """按 'a.b.c' 点路径取配置值，缺失回退 default。"""
    node: Any = cfg
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node
