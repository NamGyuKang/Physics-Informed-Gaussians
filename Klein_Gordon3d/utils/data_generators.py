import os
from utils.data_utils import *

import jax
import scipy.io

#======================== Klein-Gordon equation 3-d ========================#
#---------------------------------- PINN -----------------------------------#
@partial(jax.jit, static_argnums=(0,))
def _pinn_train_generator_klein_gordon3d(nc, k, key):
    ni, nb = nc**2, nc**2
    keys = jax.random.split(key, 13)
    # collocation points
    tc = jax.random.uniform(keys[0], (nc**3, 1), minval=0., maxval=10.)
    xc = jax.random.uniform(keys[1], (nc**3, 1), minval=-1., maxval=1.)
    yc = jax.random.uniform(keys[2], (nc**3, 1), minval=-1., maxval=1.)
    uc = klein_gordon3d_source_term(tc, xc, yc, k)
    # initial points
    ti = jnp.zeros((ni, 1))
    xi = jax.random.uniform(keys[3], (ni, 1), minval=-1., maxval=1.)
    yi = jax.random.uniform(keys[4], (ni, 1), minval=-1., maxval=1.)
    ui = klein_gordon3d_exact_u(ti, xi, yi, k)
    # boundary points (hard-coded)
    tb = [
        jax.random.uniform(keys[5], (nb, 1), minval=0., maxval=10.),
        jax.random.uniform(keys[6], (nb, 1), minval=0., maxval=10.),
        jax.random.uniform(keys[7], (nb, 1), minval=0., maxval=10.),
        jax.random.uniform(keys[8], (nb, 1), minval=0., maxval=10.)
    ]
    xb = [
        jnp.array([[-1.]]*nb),
        jnp.array([[1.]]*nb),
        jax.random.uniform(keys[9], (nb, 1), minval=-1., maxval=1.),
        jax.random.uniform(keys[10], (nb, 1), minval=-1., maxval=1.)
    ]
    yb = [
        jax.random.uniform(keys[11], (nb, 1), minval=-1., maxval=1.),
        jax.random.uniform(keys[12], (nb, 1), minval=-1., maxval=1.),
        jnp.array([[-1.]]*nb),
        jnp.array([[1.]]*nb)
    ]
    ub = []
    for i in range(4):
        ub += [klein_gordon3d_exact_u(tb[i], xb[i], yb[i], k)]
    tb = jnp.concatenate(tb)
    xb = jnp.concatenate(xb)
    yb = jnp.concatenate(yb)
    ub = jnp.concatenate(ub)
    return tc, xc, yc, uc, ti, xi, yi, ui, tb, xb, yb, ub


#---------------------------------- SPINN ----------------------------------#
@partial(jax.jit, static_argnums=(0,))
def _spinn_train_generator_klein_gordon3d(nc, k, key):
    keys = jax.random.split(key, 3)
    # collocation points
    tc = jax.random.uniform(keys[0], (nc, 1), minval=0., maxval=10.)
    xc = jax.random.uniform(keys[1], (nc, 1), minval=-1., maxval=1.)
    yc = jax.random.uniform(keys[2], (nc, 1), minval=-1., maxval=1.)
    tc_mesh, xc_mesh, yc_mesh = jnp.meshgrid(tc.ravel(), xc.ravel(), yc.ravel(), indexing='ij')
    uc = klein_gordon3d_source_term(tc_mesh, xc_mesh, yc_mesh, k)
    # initial points
    ti = jnp.zeros((1, 1))
    xi = xc
    yi = yc
    ti_mesh, xi_mesh, yi_mesh = jnp.meshgrid(ti.ravel(), xi.ravel(), yi.ravel(), indexing='ij')
    ui = klein_gordon3d_exact_u(ti_mesh, xi_mesh, yi_mesh, k)
    # boundary points (hard-coded)
    tb = [tc, tc, tc, tc]
    xb = [jnp.array([[-1.]]), jnp.array([[1.]]), xc, xc]
    yb = [yc, yc, jnp.array([[-1.]]), jnp.array([[1.]])]
    ub = []
    for i in range(4):
        tb_mesh, xb_mesh, yb_mesh = jnp.meshgrid(tb[i].ravel(), xb[i].ravel(), yb[i].ravel(), indexing='ij')
        ub += [klein_gordon3d_exact_u(tb_mesh, xb_mesh, yb_mesh, k)]
    return tc, xc, yc, uc, ti, xi, yi, ui, tb, xb, yb, ub


def generate_train_data(args, key, result_dir=None):
    eqn = args.equation
    if args.model == 'pinn':
        if eqn == 'klein_gordon3d':
            data = _pinn_train_generator_klein_gordon3d(
                args.nc, args.k, key
            )
        else:
            raise NotImplementedError
    elif args.model == 'spinn':
        if eqn == 'klein_gordon3d':
            data = _spinn_train_generator_klein_gordon3d(
                args.nc, args.k, key
            )
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return data


#----------------------- Klein-Gordon equation 3-d -------------------------#
@partial(jax.jit, static_argnums=(0, 1,))
def _test_generator_klein_gordon3d(model, nc_test, k):
    t = jnp.linspace(0, 10, nc_test)
    x = jnp.linspace(-1, 1, nc_test)
    y = jnp.linspace(-1, 1, nc_test)
    t = jax.lax.stop_gradient(t)
    x = jax.lax.stop_gradient(x)
    y = jax.lax.stop_gradient(y)
    tm, xm, ym = jnp.meshgrid(t, x, y, indexing='ij')
    u_gt = klein_gordon3d_exact_u(tm, xm, ym, k)
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
    if eqn == 'klein_gordon3d':
        data = _test_generator_klein_gordon3d(
            args.model, args.nc_test, args.k
        )
    # else:
    #     raise NotImplementedError
    return data