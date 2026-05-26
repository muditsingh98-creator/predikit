"""
Example 04 — Confidence-aware routing

Demonstrates the three on_low_confidence modes:
  "warn"     → prediction returned with _low_confidence / _confidence flags
  "raise"    → LowConfidenceError raised for the agent to handle
  "fallback" → a secondary tool is invoked instead

threshold=0.999 guarantees the low-confidence path fires for illustration.
In production, set this between 0.75 and 0.95.
"""
from pydantic import BaseModel, Field
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from predikit import LowConfidenceError, ModelTool

# ---------------------------------------------------------------------------
# Train primary (Logistic Regression) and fallback (Naive Bayes) classifiers
# ---------------------------------------------------------------------------
X, y = load_iris(return_X_y=True)
primary_clf = LogisticRegression(max_iter=200).fit(X, y)
fallback_clf = GaussianNB().fit(X, y)


class IrisInput(BaseModel):
    sepal_length: float = Field(description="Sepal length in cm")
    sepal_width:  float = Field(description="Sepal width in cm")
    petal_length: float = Field(description="Petal length in cm")
    petal_width:  float = Field(description="Petal width in cm")


fallback_tool = ModelTool(
    model=fallback_clf,
    name="iris_fallback",
    description="Naive Bayes fallback for ambiguous iris predictions.",
    input_schema=IrisInput,
    output_name="species",
    output_description="Predicted species: 0=setosa, 1=versicolor, 2=virginica",
)

# Ambiguous sample — sits between versicolor and virginica
sample = {"sepal_length": 6.3, "sepal_width": 2.5, "petal_length": 4.9, "petal_width": 1.5}

# ---------------------------------------------------------------------------
# Mode 1: warn — returns prediction + confidence flags
# ---------------------------------------------------------------------------
warn_tool = ModelTool(
    model=primary_clf,
    name="iris_warn",
    description="Classify iris, flagging uncertain predictions.",
    input_schema=IrisInput,
    output_name="species",
    output_description="Predicted species index",
    confidence_threshold=0.999,
    on_low_confidence="warn",
)

print("=== Mode: warn ===")
result = warn_tool.invoke(sample)
print(result)
if result.get("_low_confidence"):
    print(f"  → confidence {result['_confidence']:.3f} below threshold; agent should seek clarification")

# ---------------------------------------------------------------------------
# Mode 2: raise — lets the calling agent handle uncertainty explicitly
# ---------------------------------------------------------------------------
raise_tool = ModelTool(
    model=primary_clf,
    name="iris_raise",
    description="Classify iris, refusing ambiguous predictions.",
    input_schema=IrisInput,
    output_name="species",
    output_description="Predicted species index",
    confidence_threshold=0.999,
    on_low_confidence="raise",
)

print("\n=== Mode: raise ===")
try:
    raise_tool.invoke(sample)
except LowConfidenceError as e:
    print(f"  Caught LowConfidenceError: {e}")

# ---------------------------------------------------------------------------
# Mode 3: fallback — automatically routes to a secondary tool
# ---------------------------------------------------------------------------
routed_tool = ModelTool(
    model=primary_clf,
    name="iris_routed",
    description="Classify iris, routing uncertain cases to Naive Bayes.",
    input_schema=IrisInput,
    output_name="species",
    output_description="Predicted species index",
    confidence_threshold=0.999,
    on_low_confidence="fallback",
    fallback_tool=fallback_tool,
)

print("\n=== Mode: fallback ===")
result = routed_tool.invoke(sample)
print(f"  Result (served by fallback): {result}")
