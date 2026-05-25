import pytest
from pydantic import BaseModel

from predikit.coerce import coerce_inputs, coerce_value


def test_coerce_bool_strings():
    assert coerce_value("true", bool) is True
    assert coerce_value("yes", bool) is True
    assert coerce_value("false", bool) is False
    assert coerce_value("0", bool) is False


def test_coerce_bool_invalid():
    with pytest.raises(ValueError):
        coerce_value("maybe", bool)


def test_coerce_int_from_string():
    assert coerce_value("42", int) == 42
    assert coerce_value("3.7", int) == 3


def test_coerce_float_from_string():
    assert coerce_value("3.14", float) == pytest.approx(3.14)


def test_coerce_inputs_ordered_by_feature_names():
    class Schema(BaseModel):
        b: float
        a: float

    validated = Schema(b=2.0, a=1.0)
    meta = {"feature_names": ["a", "b"]}
    result = coerce_inputs(validated, meta)
    assert result == [1.0, 2.0]


def test_coerce_inputs_no_feature_names():
    class Schema(BaseModel):
        x: float
        y: float

    validated = Schema(x=3.0, y=4.0)
    meta = {"feature_names": None}
    result = coerce_inputs(validated, meta)
    assert result == [3.0, 4.0]


def test_coerce_inputs_missing_feature_raises():
    class Schema(BaseModel):
        x: float

    validated = Schema(x=1.0)
    meta = {"feature_names": ["x", "y"]}
    with pytest.raises(ValueError, match="missing"):
        coerce_inputs(validated, meta)
