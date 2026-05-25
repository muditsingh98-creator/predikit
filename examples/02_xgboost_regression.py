"""
Example 02 — XGBoost regression model wrapped as a tool.
Requires: pip install modelbridge[xgboost]
"""
import json

from pydantic import BaseModel, Field
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBRegressor
except ImportError:
    raise SystemExit("XGBoost not installed. Run: pip install modelbridge[xgboost]")

from predikit import ModelTool

# 1. Train
X, y = make_regression(n_samples=500, n_features=4, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
reg = XGBRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_train)


# 2. Define input schema
class RegressionInput(BaseModel):
    feature_1: float = Field(description="Input feature 1")
    feature_2: float = Field(description="Input feature 2")
    feature_3: float = Field(description="Input feature 3")
    feature_4: float = Field(description="Input feature 4")


# 3. Wrap
tool = ModelTool(
    model=reg,
    name="xgb_regressor",
    description="Predict a continuous value from 4 numeric features using XGBoost.",
    input_schema=RegressionInput,
    output_name="predicted_value",
    output_description="Predicted numeric output",
)

print("=== OpenAI Tool Schema ===")
print(json.dumps(tool.to_openai(), indent=2))

print("\n=== Invoke ===")
result = tool.invoke({
    "feature_1": float(X_test[0, 0]),
    "feature_2": float(X_test[0, 1]),
    "feature_3": float(X_test[0, 2]),
    "feature_4": float(X_test[0, 3]),
})
print(f"Prediction: {result}")
print(f"Actual:     {y_test[0]:.2f}")
