import pytest

from turbofan_rul.data import load_subset
from turbofan_rul.synthetic import generate


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory):
    return generate(tmp_path_factory.mktemp("cmapss"), subset="FD001", seed=0)


@pytest.fixture(scope="session")
def synthetic_data(synthetic_dir):
    return load_subset(synthetic_dir, "FD001")
