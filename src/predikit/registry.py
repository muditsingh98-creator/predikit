from __future__ import annotations
from .tool import ModelTool


class ToolRegistry:
    """Bundles multiple ModelTools and provides bulk export methods."""

    def __init__(self, tools: list[ModelTool]) -> None:
        self._tools: dict[str, ModelTool] = {t.name: t for t in tools}

    def get(self, name: str) -> ModelTool:
        if name not in self._tools:
            raise KeyError(f"No tool named '{name}'. Available: {list(self._tools)}")
        return self._tools[name]

    def to_openai(self) -> list[dict]:
        return [t.to_openai() for t in self._tools.values()]

    def to_langchain(self) -> list:
        return [t.to_langchain() for t in self._tools.values()]
