import pytest
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from predikit import ModelTool, ToolRegistry


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@pytest.fixture
def registry():
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(max_iter=200).fit(X, y)
    tool = ModelTool(
        model=clf,
        name="iris_classifier",
        description="Classify iris species",
        input_schema=IrisInput,
        output_name="species",
        output_description="Predicted species",
    )
    return ToolRegistry([tool])


def test_get_tool(registry):
    tool = registry.get("iris_classifier")
    assert tool.name == "iris_classifier"


def test_get_missing_raises(registry):
    with pytest.raises(KeyError):
        registry.get("nonexistent")


def test_to_openai_returns_list(registry):
    schemas = registry.to_openai()
    assert isinstance(schemas, list)
    assert len(schemas) == 1
    assert schemas[0]["type"] == "function"
