import sooki
import jax
import pytest


@pytest.fixture(scope="session", autouse=True)
def register_ffi_targets():
    jax.config.update("jax_enable_x64", True)

    for name, target in sooki.registrations().items():
        jax.ffi.register_ffi_target(name, target)
