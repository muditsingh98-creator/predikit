from pydantic import BaseModel, Field
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from predikit import ModelTool


class IrisInput(BaseModel):
    sepal_length: float = Field(description="Sepal length in cm")
    sepal_width: float = Field(description="Sepal width in cm")
    petal_length: float = Field(description="Petal length in cm")
    petal_width: float = Field(description="Petal width in cm")


def _make_tool() -> ModelTool:
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(max_iter=200).fit(X, y)
    return ModelTool(
        model=clf,
        name="iris_classifier",
        description="Classify iris species from petal/sepal measurements",
        input_schema=IrisInput,
        output_name="species",
        output_description="Predicted species index",
    )


def test_openai_schema_top_level_structure():
    schema = _make_tool().to_openai()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "iris_classifier"
    assert "parameters" in schema["function"]


def test_openai_schema_has_all_fields():
    props = _make_tool().to_openai()["function"]["parameters"]["properties"]
    assert set(props.keys()) == {"sepal_length", "sepal_width", "petal_length", "petal_width"}


def test_openai_schema_no_title_at_top():
    params = _make_tool().to_openai()["function"]["parameters"]
    assert "title" not in params
