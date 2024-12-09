import jax
import jax.numpy as jnp
from functools import partial
import pdb


def relative_l2(u, u_gt):
    return jnp.linalg.norm(u-u_gt) / jnp.linalg.norm(u_gt)

def mse(u, u_gt):
    return jnp.mean((u-u_gt)**2)


@partial(jax.jit, static_argnums=(0,))
def _eval3d(apply_fn, params, *test_data):
    x, y, z, u_gt = test_data
    pred = apply_fn(params, x, y, z)
    return relative_l2(pred, u_gt)


@partial(jax.jit, static_argnums=(0,))
def _eval3d_ns_pinn(apply_fn, params, *test_data):
    x, y, z, u_gt = test_data
    pred = velocity_to_vorticity_rev(apply_fn, params, x, y, z)
    return relative_l2(pred, u_gt)


@partial(jax.jit, static_argnums=(0,))
def _eval3d_ns_spinn(apply_fn, params, *test_data):
    x, y, z, u_gt = test_data
    pred = velocity_to_vorticity_fwd(apply_fn, params, x, y, z)
    return relative_l2(pred, u_gt)


def setup_eval_function(model, equation):
    dim = equation[-2:]
    if dim == '3d':
        fn = _eval3d
    else:
        raise NotImplementedError
    return fn