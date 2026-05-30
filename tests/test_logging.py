import pytest
from pydantic import BaseModel
from predikit import ModelTool
from unittest.mock import MagicMock

# 1. Define a simple schema for the test
class MockInput(BaseModel):
    feature1: float
    feature2: float

def test_verbose_logging(capsys):
    # 2. Create a fake model that mimics sklearn
    mock_model = MagicMock()
    mock_model.predict.return_value = [1]
    
    # 3. Initialize with all required arguments + verbose=True
    tool = ModelTool(
        model=mock_model, 
        name="test_tool", 
        description="testing logs",
        input_schema=MockInput,
        output_name="prediction",
        output_description="The result of the mock test",
        verbose=True
    )
    
    # 4. Run the tool using .invoke()
    tool.invoke({"feature1": 10.0, "feature2": 20.0})
    
    # 5. Check if the logs printed out to the system standard output
    captured = capsys.readouterr()
    
    # Assertions to ensure our new feature works
    assert "[predikit] Invoking tool: test_tool" in captured.out
    assert "[predikit] Prediction: 1" in captured.out
    assert "Latency" in captured.out
