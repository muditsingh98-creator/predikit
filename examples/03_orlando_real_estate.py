"""
Example 03 — Orlando Real Estate Price Predictor  (portfolio demo)

Trains an XGBoost model on synthetic Orlando-area housing data,
wraps it as an LLM-callable tool, and shows the full end-to-end flow:
schema generation → registry export → direct invocation → callable.

Requires: pip install modelbridge[xgboost]
"""
import json

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBRegressor
except ImportError:
    raise SystemExit("XGBoost not installed. Run: pip install modelbridge[xgboost]")

from predikit import ModelTool, ToolRegistry

# ---------------------------------------------------------------------------
# 1. Synthetic Orlando dataset
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
n = 1_000

sqft           = rng.integers(800, 4_000, n).astype(float)
bedrooms       = rng.integers(1, 6, n).astype(float)
bathrooms      = rng.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], n)
year_built     = rng.integers(1960, 2024, n).astype(float)
has_pool       = rng.integers(0, 2, n).astype(float)
zip_code_group = rng.integers(0, 5, n).astype(float)  # 0=rural … 4=prime

price = (
    80 * sqft
    + 15_000 * bedrooms
    + 10_000 * bathrooms
    + 500   * (year_built - 1960)
    + 20_000 * has_pool
    + 30_000 * zip_code_group
    + rng.normal(0, 15_000, n)
).clip(50_000, 1_500_000)

df = pd.DataFrame({
    "sqft":           sqft,
    "bedrooms":       bedrooms,
    "bathrooms":      bathrooms,
    "year_built":     year_built,
    "has_pool":       has_pool,
    "zip_code_group": zip_code_group,
})

X_train, X_test, y_train, y_test = train_test_split(df, price, test_size=0.2, random_state=42)

# ---------------------------------------------------------------------------
# 2. Train model
# ---------------------------------------------------------------------------
model = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# 3. Input schema — field names must match DataFrame columns exactly
# ---------------------------------------------------------------------------
class OrlandoHouseInput(BaseModel):
    sqft:           float = Field(description="Total square footage of the home")
    bedrooms:       float = Field(description="Number of bedrooms")
    bathrooms:      float = Field(description="Number of bathrooms (0.5 increments)")
    year_built:     float = Field(description="Year the home was built")
    has_pool:       float = Field(description="1 if the home has a pool, else 0")
    zip_code_group: float = Field(description="Area cluster 0–4 (0=rural, 4=prime location)")

# ---------------------------------------------------------------------------
# 4. Wrap as ModelTool and register
# ---------------------------------------------------------------------------
price_tool = ModelTool(
    model=model,
    name="orlando_home_price",
    description=(
        "Estimate the sale price of a residential property in the Orlando, FL metro area "
        "based on its characteristics. Returns predicted price in USD."
    ),
    input_schema=OrlandoHouseInput,
    output_name="estimated_price_usd",
    output_description="Predicted home sale price in US dollars",
)

registry = ToolRegistry([price_tool])

# ---------------------------------------------------------------------------
# 5. Show OpenAI function schema
# ---------------------------------------------------------------------------
print("=== OpenAI Function Schema ===")
print(json.dumps(registry.to_openai(), indent=2))

# ---------------------------------------------------------------------------
# 6. Run a prediction
# ---------------------------------------------------------------------------
sample = {
    "sqft":           2_200.0,
    "bedrooms":       3.0,
    "bathrooms":      2.0,
    "year_built":     2005.0,
    "has_pool":       1.0,
    "zip_code_group": 3.0,
}

print("\n=== Sample Prediction ===")
print(f"Input:  {sample}")
result = price_tool.invoke(sample)
print(f"Output: {result}")
print(f"\nEstimated price: ${result['estimated_price_usd']:,.0f}")

# ---------------------------------------------------------------------------
# 7. Same call via plain callable (simulates how an LLM agent would invoke it)
# ---------------------------------------------------------------------------
print("\n=== Via Plain Callable (LLM-style) ===")
fn = price_tool.to_callable()
result2 = fn(**sample)
print(f"Result: {result2}")
