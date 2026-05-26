from __future__ import annotations
from collections import Counter
from typing import Any, Callable

from .tool import ModelTool
from .exporters.openai import to_openai_schema
from .exporters.langchain import to_langchain_tool


class ModelEnsemble:
    """Calls multiple ModelTools and reconciles their outputs into one dict."""

    def __init__(
        self,
        tools: list[ModelTool],
        name: str,
        description: str,
        strategy: str = "collect",
    ) -> None:
        if not tools:
            raise ValueError("ModelEnsemble requires at least one tool.")
        self.tools = tools
        self.name = name
        self.description = description
        self.strategy = strategy
        self.input_schema = tools[0].input_schema
        self.output_description = tools[0].output_description

    @property
    def output_name(self) -> str:
        return self.tools[0].output_name

    def invoke(self, input_dict: dict) -> dict:
        """Call all tools and reconcile outputs using the configured strategy."""
        results = [t.invoke(input_dict) for t in self.tools]

        if self.strategy == "collect":
            merged: dict = {}
            for r in results:
                merged.update(r)
            return merged

        output_name = self.tools[0].output_name
        values = [r[output_name] for r in results]

        if self.strategy == "mean":
            numeric = [float(v) for v in values]
            return {output_name: sum(numeric) / len(numeric)}

        if self.strategy == "vote":
            return {output_name: Counter(values).most_common(1)[0][0]}

        raise ValueError(f"Unknown strategy '{self.strategy}'. Use 'collect', 'mean', or 'vote'.")

    def to_openai(self) -> dict:
        return to_openai_schema(self)

    def to_langchain(self):
        return to_langchain_tool(self)

    def to_callable(self) -> Callable[..., dict]:
        def _fn(**kwargs) -> dict:
            return self.invoke(kwargs)
        _fn.__name__ = self.name
        _fn.__doc__ = self.description
        return _fn
