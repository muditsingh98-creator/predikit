from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..tool import ModelTool


def to_langchain_tool(tool: ModelTool):
    """Convert a ModelTool to a LangChain StructuredTool."""
    try:
        from langchain_core.tools import StructuredTool
    except ImportError:
        raise ImportError(
            "langchain-core is required. Install with: pip install predikit[langchain]"
        )

    def _run(**kwargs) -> dict:
        return tool.invoke(kwargs)

    return StructuredTool.from_function(
        func=_run,
        name=tool.name,
        description=tool.description,
        args_schema=tool.input_schema,
    )
