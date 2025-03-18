import pytest
import os

@pytest.fixture
def input_path():
    # Get the absolute path to the test data folder
    return os.path.join(os.path.dirname(__file__), 'input')
