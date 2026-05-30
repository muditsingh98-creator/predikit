from __future__ import annotations
import time  # <--- Added for execution timing
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel

from .introspect import introspect
from .coerce import coerce_inputs, coerce_value
from .exporters.openai import to_openai_schema
from .exporters.langchain import to_langchain_tool
from .exceptions import LowConfidenceError

_VALID_ON_LOW_CONFIDENCE = {"warn", "raise", "fallback"}
_SCALAR_TYPES = (bool, int, float, str)

class ModelTool:
    """Wraps a fitted sklearn-compatible model as an LLM-callable tool."""

    def __init__(
        self,
        model: Any,
        name: str,
        description: str,
        input_schema: type[BaseModel],
        output_name: str,
        output_description: str,
        confidence_threshold: float | None = None,
        on_low_confidence: str = "warn",
        fallback_tool: ModelTool | None = None,
        verbose: bool = False,  # <--- Added verbose argument
    ) -> None:
        if on_low_confidence not in _VALID_ON_LOW_CONFIDENCE:
            raise ValueError(f"on_low_confidence must be one of {_VALID_ON_LOW_CONFIDENCE}, got {on_low_confidence!r}")
        self.model = model
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.output_name = output_name
        self.output_description = output_description
        self._meta = introspect(model)
        self.confidence_threshold = confidence_threshold
        self.on_low_confidence = on_low_confidence
        self.fallback_tool = fallback_tool
        self.verbose = verbose  # <--- Added storage of verbose setting

    def invoke(self, input_dict: dict) -> dict:
        """Validate inputs, run prediction, return {output_name: value}."""
        # 1. Logging Input if verbose
        if self.verbose:
            print(f"[predikit] Invoking tool: {self.name}")
            print(f"[predikit] Input dict: {input_dict}")

        pre_coerced = self._pre_coerce(input_dict)
        try:
            validated = self.input_schema(**pre_coerced)
        except Exception as exc:
            raise ValueError(f"Input validation failed for '{self.name}': {exc}") from exc

        features = coerce_inputs(validated, self._meta)
        X = self._to_array(features)
        
        # 2. Start timing the prediction
        start_time = time.perf_counter()
        prediction = self.model.predict(X)[0]
        duration = (time.perf_counter() - start_time) * 1000

        if hasattr(prediction, "item"):
            prediction = prediction.item()

        # 3. Logging Output and Latency if verbose
        if self.verbose:
            print(f"[predikit] Prediction: {prediction}")
            print(f"[predikit] Latency: {duration:.2f}ms")

        if (
            self.confidence_threshold is not None
            and self._meta["task"] == "classification"
            and hasattr(self.model, "predict_proba")
        ):
            proba = self.model.predict_proba(X)[0]
            confidence = float(max(proba))
            if confidence < self.confidence_threshold:
                return self._handle_low_confidence(input_dict, prediction, confidence)

        return {self.output_name: prediction}

    # ... remaining methods (to_openai, to_langchain, etc.) stay exactly the same
