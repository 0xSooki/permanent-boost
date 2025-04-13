import sooki
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

for name, target in sooki.registrations().items():
    jax.ffi.register_ffi_target(name, target)

gpu = False
gpu_targets = {}
if hasattr(sooki, 'gpu_ops'):
    try:
        gpu_targets = sooki.gpu_ops.foo()
        for name, target in gpu_targets.items():
            jax.ffi.register_ffi_target(name, target, platform="CUDA")
            gpu = True
    except (ImportError, AttributeError) as e:
        print(f"GPU support initialization failed: {e}")
        gpu = False
else:
    print("No GPU module found. Continuing with CPU support only.")


@partial(jax.custom_vjp)
def perm(A, rows, cols):
    if A.dtype != jnp.complex128:
        raise ValueError("Only the float32 dtype is implemented by rms_norm")

    out_type = jax.ShapeDtypeStruct((), A.dtype)

    def impl(target_name):
        return lambda: jax.ffi.ffi_call(
            target_name,
            out_type,
            vmap_method="broadcast_all",
        )(A, rows, cols)

    return jax.lax.platform_dependent(
        cpu=impl("perm"),
        cuda=impl("dperm")
    )


def perm_fwd(A, rows, cols):
    def impl(target_name):
        return lambda: jax.ffi.ffi_call(
            target_name,
            (
                jax.ShapeDtypeStruct((), A.dtype),
                jax.ShapeDtypeStruct((), A.dtype),
            ),
            vmap_method="broadcast_all",
        )(A, rows, cols)
    y, res = jax.lax.platform_dependent(
        cpu=impl("perm_fwd"),
        cuda=impl("dperm_fwd")
    )
    return y, (res, A, rows, cols)


def perm_bwd(res, _):
    res, A, rows, cols = res

    def impl(target_name):
        return lambda: (jax.ffi.ffi_call(
            target_name,
            jax.ShapeDtypeStruct(A.shape, A.dtype),
            vmap_method="broadcast_all",
        )(res, A, rows, cols), None, None)
    
    return jax.lax.platform_dependent(
        cpu=impl("perm_bwd"),
        cuda=impl("dperm_bwd")
    )

perm.defvjp(perm_fwd, perm_bwd)


# @partial(jax.custom_vjp)
# def perm(A, rows, cols):
#     if A.dtype != jnp.complex128:
#         raise ValueError("Only the float32 dtype is implemented by rms_norm")

#     out_type = jax.ShapeDtypeStruct((), A.dtype)

#     return jax.ffi.ffi_call(
#         "perm",
#         out_type,
#         vmap_method="broadcast_all",
#     )(A, rows, cols)


# def perm_fwd(A, rows, cols):
#     y, res = jax.ffi.ffi_call(
#         "perm_fwd",
#         (
#             jax.ShapeDtypeStruct((), A.dtype),
#             jax.ShapeDtypeStruct((), A.dtype),
#         ),
#         vmap_method="broadcast_all",
#     )(A, rows, cols)
#     return y, (res, A, rows, cols)


# def perm_bwd(res, _):
#     res, A, rows, cols = res
#     # assert res.shape == ct.shape[:-1]
#     # assert A.shape == ct.shape
#     return (
#         jax.ffi.ffi_call(
#             "perm_bwd",
#             jax.ShapeDtypeStruct((A.shape), A.dtype),
#             vmap_method="broadcast_all",
#         )(res, A, rows, cols), None, None
#     )


# perm.defvjp(perm_fwd, perm_bwd)
