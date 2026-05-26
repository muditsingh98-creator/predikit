"""
Example 05 — Multi-model ensemble

Demonstrates ModelEnsemble with all three strategies:
  "collect" → merges all tool outputs into one dict (tools can have different output names)
  "mean"    → averages numeric outputs (tools must share output_name)
  "vote"    → majority class vote (tools must share output_name)

Requires: pip install predikit[xgboost]
"""
import json

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBRegressor
except ImportError:
    raise SystemExit("XGBoost not installed. Run: pip install predikit[xgboost]")

from predikit import ModelEnsemble, ModelTool, ToolRegistry

# ---------------------------------------------------------------------------
# Synthetic Orlando dataset (same as example 03)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
n = 1_000
sqft           = rng.integers(800, 4_000, n).astype(float)
bedrooms       = rng.integers(1, 6, n).astype(float)
bathrooms      = rng.choice([1.0, 1.5, 2.0, 2.5, 3.0], n)
year_built     = rng.integers(1960, 2024, n).astype(float)
has_pool       = rng.integers(0, 2, n).astype(float)
zip_code_group = rng.integers(0, 5, n).astype(float)
price = (
    80 * sqft + 15_000 * bedrooms + 10_000 * bathrooms
    + 500 * (year_built - 1960) + 20_000 * has_pool
    + 30_000 * zip_code_group + rng.normal(0, 15_000, n)
).clip(50_000, 1_500_000)

df = pd.DataFrame({
    "sqft": sqft, "bedrooms": bedrooms, "bathrooms": bathrooms,
    "year_built": year_built, "has_pool": has_pool, "zip_code_group": zip_code_group,
})
X_train, X_test, y_train, _ = train_test_split(df, price, test_size=0.2, random_state=42)


class HouseInput(BaseModel):
    sqft:           float = Field(description="Square footage")
    bedrooms:       float = Field(description="Number of bedrooms")
    bathrooms:      float = Field(description="Number of bathrooms")
    year_built:     float = Field(description="Year built")
    has_pool:       float = Field(description="1 if has pool, else 0")
    zip_code_group: float = Field(description="Area cluster 0–4")


sample = {
    "sqft": 2200.0, "bedrooms": 3.0, "bathrooms": 2.0,
    "year_built": 2005.0, "has_pool": 1.0, "zip_code_group": 3.0,
}

# ---------------------------------------------------------------------------
# Strategy 1: "collect" — different output names, results merged
# ---------------------------------------------------------------------------
xgb_tool = ModelTool(
    model=XGBRegressor(n_estimators=100, random_state=42).fit(X_train, y_train),
    name="xgb_price", description="XGBoost price estimate",
    input_schema=HouseInput, output_name="xgb_price_usd", output_description="XGBoost price",
)
lr_tool = ModelTool(
    model=LinearRegression().fit(X_train, y_train),
    name="lr_price", description="Linear regression price estimate",
    input_schema=HouseInput, output_name="lr_price_usd", output_description="LR price",
)

collect_ensemble = ModelEnsemble(
    tools=[xgb_tool, lr_tool],
    name="price_comparison",
    description="Returns both XGBoost and linear regression price estimates.",
    strategy="collect",
)

print("=== Strategy: collect ===")
result = collect_ensemble.invoke(sample)
print(result)
print(f"  XGBoost:  ${result['xgb_price_usd']:,.0f}")
print(f"  Linear:   ${result['lr_price_usd']:,.0f}")

# ---------------------------------------------------------------------------
# Strategy 2: "mean" — two XGBoost variants, same output_name, average them
# ---------------------------------------------------------------------------
mean_tools = [
    ModelTool(
        model=XGBRegressor(n_estimators=100, random_state=seed).fit(X_train, y_train),
        name=f"xgb_{seed}", description=f"XGBoost seed={seed}",
        input_schema=HouseInput, output_name="price_usd", output_description="price",
    )
    for seed in [0, 1]
]

mean_ensemble = ModelEnsemble(
    tools=mean_tools,
    name="averaged_price",
    description="Ensemble price: mean of two XGBoost models with different seeds.",
    strategy="mean",
)

print("\n=== Strategy: mean ===")
result = mean_ensemble.invoke(sample)
print(f"  Averaged price: ${result['price_usd']:,.0f}")

# ---------------------------------------------------------------------------
# Strategy 3: "vote" — three logistic regressors, majority class wins
# ---------------------------------------------------------------------------
X_iris, y_iris = load_iris(return_X_y=True)


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width:  float
    petal_length: float
    petal_width:  float


clf_tools = [
    ModelTool(
        model=LogisticRegression(max_iter=200, C=c).fit(X_iris, y_iris),
        name=f"clf_C{c}", description=f"LR C={c}",
        input_schema=IrisInput, output_name="species", output_description="species",
    )
    for c in [0.1, 1.0, 10.0]
]

vote_ensemble = ModelEnsemble(
    tools=clf_tools,
    name="iris_vote",
    description="Majority vote across three logistic regression classifiers.",
    strategy="vote",
)

iris_sample = {"sepal_length": 6.3, "sepal_width": 2.5, "petal_length": 4.9, "petal_width": 1.5}
print("\n=== Strategy: vote ===")
print(vote_ensemble.invoke(iris_sample))

# ---------------------------------------------------------------------------
# ToolRegistry with ensembles
# ---------------------------------------------------------------------------
print("\n=== OpenAI schemas (registry with ensemble) ===")
registry = ToolRegistry(tools=[xgb_tool, lr_tool], ensembles=[collect_ensemble])
schemas = registry.to_openai()
print(json.dumps([s["function"]["name"] for s in schemas], indent=2))
print(f"\n{len(schemas)} total schemas (2 tools + 1 ensemble)")
