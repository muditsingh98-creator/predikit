from typing import Any


def introspect(model: Any) -> dict:
    """Extract metadata from a fitted sklearn-compatible estimator."""
    meta: dict = {}

    meta["feature_names"] = (
        list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None
    )
    meta["n_features"] = (
        int(model.n_features_in_) if hasattr(model, "n_features_in_") else None
    )

    if hasattr(model, "classes_"):
        meta["task"] = "classification"
        meta["classes"] = list(model.classes_)
    else:
        meta["task"] = "regression"
        meta["classes"] = None

    return meta
