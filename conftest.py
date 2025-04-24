import sooki
import jax
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--platform", action="store", default="cpu", help="Choose platform: cpu or gpu"
    )


def pytest_configure(config):
    platform = config.getoption("--platform")
    jax.config.update("jax_platform_name", platform)


@pytest.fixture(scope="session", autouse=True)
def register_ffi_targets():
    jax.config.update("jax_enable_x64", True)

    for name, target in sooki.registrations().items():
        jax.ffi.register_ffi_target(name, target)
