import os
from utils.data_utils import *

import jax
import scipy.io


#======================== Flow-Mixing 3-d ========================#
#----------------------------- PINN ------------------------------#
@partial(jax.jit, static_argnums=(0,))
def _pinn_train_generator_flow_mixing3d(nc, v_max, key):
    ni, nb = nc**2, nc**2

    keys = jax.random.split(key, 13)
    # collocation points
    tc = jax.random.uniform(keys[0], (nc**3, 1), minval=0., maxval=4.)
    xc = jax.random.uniform(keys[1], (nc**3, 1), minval=-4., maxval=4.)
    yc = jax.random.uniform(keys[2], (nc**3, 1), minval=-4., maxval=4.)
    _, a, b = flow_mixing3d_params(tc, xc, yc, v_max, require_ab=True)

    # initial points
    ti = jnp.zeros((ni, 1))
    xi = jax.random.uniform(keys[3], (ni, 1), minval=-4., maxval=4.)
    yi = jax.random.uniform(keys[4], (ni, 1), minval=-4., maxval=4.)
    omega_i, _, _ = flow_mixing3d_params(ti, xi, yi, v_max)
    ui = flow_mixing3d_exact_u(ti, xi, yi, omega_i)

    # boundary points (hard-coded)
    tb = [
        jax.random.uniform(keys[5], (nb, 1), minval=0., maxval=4.),
        jax.random.uniform(keys[6], (nb, 1), minval=0., maxval=4.),
        jax.random.uniform(keys[7], (nb, 1), minval=0., maxval=4.),
        jax.random.uniform(keys[8], (nb, 1), minval=0., maxval=4.)
    ]
    xb = [
        jnp.array([[-4.]]*nb),
        jnp.array([[4.]]*nb),
        jax.random.uniform(keys[9], (nb, 1), minval=-4., maxval=4.),
        jax.random.uniform(keys[10], (nb, 1), minval=-4., maxval=4.)
    ]
    yb = [
        jax.random.uniform(keys[11], (nb, 1), minval=-4., maxval=4.),
        jax.random.uniform(keys[12], (nb, 1), minval=-4., maxval=4.),
        jnp.array([[-4.]]*nb),
        jnp.array([[4.]]*nb)
    ]
    ub = []
    for i in range(4):
        omega_b, _, _ = flow_mixing3d_params(tb[i], xb[i], yb[i], v_max)
        ub += [flow_mixing3d_exact_u(tb[i], xb[i], yb[i], omega_b)]
    tb = jnp.concatenate(tb)
    xb = jnp.concatenate(xb)
    yb = jnp.concatenate(yb)
    ub = jnp.concatenate(ub)
    return tc, xc, yc, ti, xi, yi, ui, tb, xb, yb, ub, a, b


#----------------------------- SPINN -----------------------------#
@partial(jax.jit, static_argnums=(0,))
def _spinn_train_generator_flow_mixing3d(nc, v_max, key):
    keys = jax.random.split(key, 3)
    # collocation points
    tc = jax.random.uniform(keys[0], (nc, 1), minval=0., maxval=4.)
    xc = jax.random.uniform(keys[1], (nc, 1), minval=-4., maxval=4.)
    yc = jax.random.uniform(keys[2], (nc, 1), minval=-4., maxval=4.)
    tc_mesh, xc_mesh, yc_mesh = jnp.meshgrid(tc.ravel(), xc.ravel(), yc.ravel(), indexing='ij')

    _, a, b = flow_mixing3d_params(tc_mesh, xc_mesh, yc_mesh, v_max, require_ab=True)

    # initial points
    ti = jnp.zeros((1, 1))
    xi = xc
    yi = yc
    ti_mesh, xi_mesh, yi_mesh = jnp.meshgrid(ti.ravel(), xi.ravel(), yi.ravel(), indexing='ij')
    omega_i, _, _ = flow_mixing3d_params(ti_mesh, xi_mesh, yi_mesh, v_max)
    ui = flow_mixing3d_exact_u(ti_mesh, xi_mesh, yi_mesh, omega_i)
    # boundary points (hard-coded)
    tb = [tc, tc, tc, tc]
    xb = [jnp.array([[-4.]]), jnp.array([[4.]]), xc, xc]
    yb = [yc, yc, jnp.array([[-4.]]), jnp.array([[4.]])]
    ub = []
    for i in range(4):
        tb_mesh, xb_mesh, yb_mesh = jnp.meshgrid(tb[i].ravel(), xb[i].ravel(), yb[i].ravel(), indexing='ij')
        omega_b, _, _ = flow_mixing3d_params(tb_mesh, xb_mesh, yb_mesh, v_max)
        ub += [flow_mixing3d_exact_u(tb_mesh, xb_mesh, yb_mesh, omega_b)]
    return tc, xc, yc, ti, xi, yi, ui, tb, xb, yb, ub, a, b


def generate_train_data(args, key, result_dir=None):
    eqn = args.equation
    if args.model == 'pinn':
        if eqn == 'flow_mixing3d':
            data = _pinn_train_generator_flow_mixing3d(
                args.nc, args.vmax, key
            )
        else:
            raise NotImplementedError
    elif args.model == 'spinn':
        if eqn == 'flow_mixing3d':
            data = _spinn_train_generator_flow_mixing3d(
                args.nc, args.vmax, key
            )
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return data


#----------------------- Flow-Mixing 3-d -------------------------#
@partial(jax.jit, static_argnums=(0, 1,))
def _test_generator_flow_mixing3d(model, nc_test, v_max):
    t = jnp.linspace(0, 4, nc_test)
    x = jnp.linspace(-4, 4, nc_test)
    y = jnp.linspace(-4, 4, nc_test)
    t = jax.lax.stop_gradient(t)
    x = jax.lax.stop_gradient(x)
    y = jax.lax.stop_gradient(y)
    tm, xm, ym = jnp.meshgrid(t, x, y, indexing='ij')

    omega, _, _ = flow_mixing3d_params(tm, xm, ym, v_max)
    u_gt = flow_mixing3d_exact_u(tm, xm, ym, omega)

    if model == 'pinn':
        t = tm.reshape(-1, 1)
        x = xm.reshape(-1, 1)
        y = ym.reshape(-1, 1)
        u_gt = u_gt.reshape(-1, 1)
    else:
        t = t.reshape(-1, 1)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
    return t, x, y, u_gt


def generate_test_data(args, result_dir):
    eqn = args.equation
    if eqn == 'flow_mixing3d':
        data = _test_generator_flow_mixing3d(
            args.model, args.nc_test, args.vmax
        )
    # else:
    #     raise NotImplementedError
    return data