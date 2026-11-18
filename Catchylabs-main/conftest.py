import pytest
from gemini_pacer import pacer

collect_ignore = ["scripts"]


@pytest.fixture(autouse=True)
def _reset_gemini_pacer():
    pacer.reset()
    yield
    pacer.reset()
