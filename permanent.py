import sooki
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


for name, target in sooki.registrations().items():
    jax.ffi.register_ffi_target(name, target)


@partial(jax.custom_vjp)
def perm(A, rows, cols):
    if A.dtype != jnp.complex128:
        raise ValueError("Only the float32 dtype is implemented by rms_norm")

    out_type = jax.ShapeDtypeStruct((), A.dtype)

    return jax.ffi.ffi_call(
        "perm",
        out_type,
        vmap_method="broadcast_all",
    )(A, rows, cols)


def perm_fwd(A, rows, cols):
    y, res = jax.ffi.ffi_call(
        "perm_fwd",
        (
            jax.ShapeDtypeStruct((), A.dtype),
            jax.ShapeDtypeStruct((), A.dtype),
        ),
        vmap_method="broadcast_all",
    )(A, rows, cols)
    return y, (res, A, rows, cols)


def perm_bwd(res, ct):
    res, A, rows, cols = res
    # assert res.shape == ct.shape[:-1]
    # assert A.shape == ct.shape
    return (
        jax.ffi.ffi_call(
            "perm_bwd",
            jax.ShapeDtypeStruct((A.shape), ct.dtype),
            vmap_method="broadcast_all",
        )(res, A, rows, cols, ct), None, None
    )


perm.defvjp(perm_fwd, perm_bwd)
