"""Load only the tracker; optional voice dependencies are not needed."""

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[2] / "src/tau2/user/goal_tracking.py"


@pytest.fixture
def tracking():
    assert MODULE_PATH.exists(), "UGST tracker has not been implemented"
    spec = importlib.util.spec_from_file_location(
        "goal_tracking_under_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
