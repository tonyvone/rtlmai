import numpy as np
import pytest

from rtlm_n.bases.basis import build_basis
from rtlm_n.benchmarks.data import make_world
from rtlm_n.core.hashing import KeyRegistry


@pytest.fixture(scope="session")
def world():
    return make_world(seed=0)


@pytest.fixture(scope="session")
def basis(world):
    return build_basis(world.small, "random-orthogonal", seed=0, ru=4, rv=8)


@pytest.fixture()
def keys():
    return KeyRegistry()


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)
