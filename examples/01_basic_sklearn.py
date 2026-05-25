"""Example 01 — Basic sklearn classifier wrapped as an OpenAI tool."""
import json

from pydantic import BaseModel, Field
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from predikit import ModelTool

# 1. Train
X, y = load_iris(return_X_y=True)
clf = LogisticRegression(max_iter=200).fit(X, y)


# 2. Define input schema
class IrisInput(BaseModel):
    sepal_length: float = Field(description="Sepal length in cm")
    sepal_width: float = Field(description="Sepal width in cm")
    petal_length: float = Field(description="Petal length in cm")
    petal_width: float = Field(description="Petal width in cm")


# 3. Wrap
tool = ModelTool(
    model=clf,
    name="classify_iris",
    description="Classify an iris flower as setosa (0), versicolor (1), or virginica (2).",
    input_schema=IrisInput,
    output_name="species",
    output_description="Predicted species index: 0=setosa, 1=versicolor, 2=virginica",
)

# 4. OpenAI schema
print("=== OpenAI Tool Schema ===")
print(json.dumps(tool.to_openai(), indent=2))

# 5. Direct invocation
print("\n=== Direct Invocation ===")
result = tool.invoke({"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2})
print(f"Result: {result}")

# 6. Plain callable
fn = tool.to_callable()
result2 = fn(sepal_length=6.3, sepal_width=3.3, petal_length=6.0, petal_width=2.5)
print(f"Via callable: {result2}")
