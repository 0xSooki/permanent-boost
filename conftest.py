import sooki
import jax
import pytest

def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true", help="Run tests using GPU backend")
    parser.addoption("--cpu", action="store_true", help="Run tests using CPU backend")

def pytest_configure(config):
    if config.getoption("--gpu"):
        jax.config.update("jax_platform_name", "gpu")
    elif config.getoption("--cpu"):
        jax.config.update("jax_platform_name", "cpu")

@pytest.fixture(scope="session", autouse=True)
def register_ffi_targets():
    jax.config.update("jax_enable_x64", True)

    for name, target in sooki.registrations().items():
        jax.ffi.register_ffi_target(name, target)
