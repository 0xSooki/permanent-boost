from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import sooki

jax.config.update("jax_enable_x64", True)

for name, target in sooki.gpu_ops.foo().items():
    print(name, target)
    jax.ffi.register_ffi_target(name, target, platform="gpu")

def foo_fwd(a, b):
  assert a.dtype == jnp.complex128
  assert a.shape == b.shape
  assert a.dtype == b.dtype
  n = np.prod(a.shape).astype(np.uint64)
  out_type = jax.ShapeDtypeStruct(a.shape, a.dtype)
  c, b_plus_1 = jax.ffi.ffi_call("foo_fwd", (out_type, out_type))(a, b, n=n)
  return c, (a, b_plus_1)


def foo_bwd(res, c_grad):
  a, b_plus_1 = res
  assert c_grad.dtype == jnp.float32
  assert c_grad.shape == a.shape
  assert a.shape == b_plus_1.shape
  assert c_grad.dtype == a.dtype
  assert a.dtype == b_plus_1.dtype
  n = np.prod(a.shape).astype(np.uint64)
  out_type = jax.ShapeDtypeStruct(a.shape, a.dtype)
  return jax.ffi.ffi_call("foo_bwd", (out_type, out_type))(c_grad, a, b_plus_1,
                          n=n)


@jax.custom_vjp
def foo(a, b):
  c, _ = foo_fwd(a, b)
  return c


foo.defvjp(foo_fwd, foo_bwd)

#create a complex128 example

a = jnp.array([[1+2j, 2+3j], [3+4j, 4+5j]], dtype=jnp.complex128)
b = jnp.array([[5+6j, 6+7j], [7+8j, 8+9j]], dtype=jnp.complex128)
c = foo(a, b)
print(c)


@partial(jax.custom_vjp)
def perm(A, rows, cols):
  if A.dtype != jnp.complex128:
      raise ValueError("Only the float32 dtype is implemented by rms_norm")

  out_type = jax.ShapeDtypeStruct((), A.dtype)

  return jax.ffi.ffi_call(
      "permm",
      out_type,
      vmap_method="broadcast_all",
  )(A, rows, cols)


def perm_fwd():
  pass

def perm_bwd():
  pass

perm.defvjp(perm_fwd, perm_bwd)



interferometer = jnp.array(
    [
        [
            0.62113733 - 0.01959968j,
            -0.15627468 - 0.0772489j,
            -0.2705819 - 0.18997122j,
            0.26504798 - 0.30838768j,
            0.03372169 - 0.11154586j,
            0.15278476 + 0.52137824j,
        ],
        [
            -0.1776024 - 0.21000195j,
            0.18950753 + 0.20741494j,
            -0.15537846 + 0.19161071j,
            0.07400899 - 0.37578572j,
            -0.44458249 - 0.0047501j,
            -0.62212719 + 0.23055313j,
        ],
        [
            -0.05572001 - 0.20287464j,
            0.22359337 + 0.30693557j,
            -0.13719319 + 0.23245719j,
            0.1102451 + 0.02659467j,
            0.81942653 + 0.04327346j,
            -0.17215559 + 0.15114287j,
        ],
        [
            -0.24319645 - 0.44143551j,
            -0.50022937 - 0.08513718j,
            0.07671116 - 0.05858231j,
            0.0679656 + 0.52109972j,
            -0.0482276 - 0.12736588j,
            -0.11768435 + 0.41307881j,
        ],
        [
            -0.29469977 - 0.20027018j,
            0.22135149 - 0.02983563j,
            -0.18587346 - 0.83950064j,
            -0.21606625 - 0.14975436j,
            0.11702974 - 0.02297493j,
            -0.01552763 + 0.01646485j,
        ],
        [
            -0.29741767 + 0.15644426j,
            -0.61959257 - 0.23497653j,
            0.07397837 + 0.05367843j,
            -0.05838964 - 0.57132173j,
            0.28736069 - 0.00798998j,
            -0.13763068 - 0.09058005j,
        ],
    ], dtype=jnp.complex128
)

input = output = jnp.ones(6, dtype=jnp.uint32)
print(np.sum(interferometer))
print(perm(interferometer, input, output))
