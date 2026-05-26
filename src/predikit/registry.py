from __future__ import annotations
from .tool import ModelTool
from .ensemble import ModelEnsemble


class ToolRegistry:
    """Bundles multiple ModelTools (and optional ModelEnsembles) for bulk export."""

    def __init__(
        self,
        tools: list[ModelTool],
        ensembles: list[ModelEnsemble] | None = None,
    ) -> None:
        self._tools: dict[str, ModelTool] = {t.name: t for t in tools}
        self._ensembles: dict[str, ModelEnsemble] = {e.name: e for e in (ensembles or [])}

    def get(self, name: str) -> ModelTool:
        if name not in self._tools:
            raise KeyError(f"No tool named '{name}'. Available: {list(self._tools)}")
        return self._tools[name]

    def to_openai(self) -> list[dict]:
        return (
            [t.to_openai() for t in self._tools.values()]
            + [e.to_openai() for e in self._ensembles.values()]
        )

    def to_langchain(self) -> list:
        return (
            [t.to_langchain() for t in self._tools.values()]
            + [e.to_langchain() for e in self._ensembles.values()]
        )
