from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression

from predikit.introspect import introspect


def test_classifier_metadata():
    X, y = load_iris(return_X_y=True, as_frame=True)
    clf = LogisticRegression(max_iter=200).fit(X, y)
    meta = introspect(clf)
    assert meta["task"] == "classification"
    assert meta["classes"] == [0, 1, 2]
    assert meta["feature_names"] == list(X.columns)
    assert meta["n_features"] == 4


def test_regressor_metadata():
    X, y = load_iris(return_X_y=True, as_frame=True)
    reg = LinearRegression().fit(X, y)
    meta = introspect(reg)
    assert meta["task"] == "regression"
    assert meta["classes"] is None
    assert meta["feature_names"] == list(X.columns)


def test_no_feature_names():
    X, y = load_iris(return_X_y=True)
    reg = LinearRegression().fit(X, y)
    meta = introspect(reg)
    assert meta["feature_names"] is None
    assert meta["n_features"] == 4
