import tempfile

import pytest


def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true", default=False, help="run tests that require a GPU")
    parser.addoption("--hub", action="store_true", default=False, help="run tests that require converted models")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--gpu"):
        skip_gpu = pytest.mark.skip(reason="need --gpu option to run")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


@pytest.fixture(scope="module")
def temp_dir():
    with tempfile.TemporaryDirectory(dir=".") as d:
        yield d
