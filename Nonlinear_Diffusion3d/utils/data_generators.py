import os

import jax
import jax.numpy as jnp
import scipy.io
from functools import partial

#========================== diffusion equation 3-d =========================#
#---------------------------------- PINN -----------------------------------#
@partial(jax.jit, static_argnums=(0,))
def _pinn_train_generator_diffusion3d(nc, key):
    keys = jax.random.split(key, 13)
    ni, nb = nc**2, nc**2

    # colocation points
    tc = jax.random.uniform(keys[0], (nc**3, 1), minval=0., maxval=1.)
    xc = jax.random.uniform(keys[1], (nc**3, 1), minval=-1., maxval=1.)
    yc = jax.random.uniform(keys[2], (nc**3, 1), minval=-1., maxval=1.)
    # initial points
    ti = jnp.zeros((ni, 1))
    xi = jax.random.uniform(keys[3], (ni, 1), minval=-1., maxval=1.)
    yi = jax.random.uniform(keys[4], (ni, 1), minval=-1., maxval=1.)
    ui = 0.25 * jnp.exp(-((xi - 0.3)**2 + (yi - 0.2)**2) / 0.1) + \
         0.4 * jnp.exp(-((xi + 0.5)**2 + (yi + 0.1)**2) * 15) + \
         0.3 * jnp.exp(-(xi**2 + (yi + 0.5)**2) * 20)
    # boundary points (hard-coded)
    tb = [
        jax.random.uniform(keys[5], (nb, 1), minval=0., maxval=1.),
        jax.random.uniform(keys[6], (nb, 1), minval=0., maxval=1.),
        jax.random.uniform(keys[7], (nb, 1), minval=0., maxval=1.),
        jax.random.uniform(keys[8], (nb, 1), minval=0., maxval=1.)
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
    tb = jnp.concatenate(tb)
    xb = jnp.concatenate(xb)
    yb = jnp.concatenate(yb)
    return tc, xc, yc, ti, xi, yi, ui, tb, xb, yb


#---------------------------------- SPINN ----------------------------------#
@partial(jax.jit, static_argnums=(0,))
def _spinn_train_generator_diffusion3d(nc, key):
    keys = jax.random.split(key, 3)
    # colocation points
    tc = jax.random.uniform(keys[0], (nc, 1), minval=0., maxval=1.)
    xc = jax.random.uniform(keys[1], (nc, 1), minval=-1., maxval=1.)
    yc = jax.random.uniform(keys[2], (nc, 1), minval=-1., maxval=1.)
    # initial points
    ti = jnp.zeros((1, 1))
    xi = xc
    yi = yc
    xi_mesh, yi_mesh = jnp.meshgrid(xi.ravel(), yi.ravel(), indexing='ij')
    ui = 0.25 * jnp.exp(-((xi_mesh - 0.3)**2 + (yi_mesh - 0.2)**2) / 0.1) + \
         0.4 * jnp.exp(-((xi_mesh + 0.5)**2 + (yi_mesh + 0.1)**2) * 15) + \
         0.3 * jnp.exp(-(xi_mesh**2 + (yi_mesh + 0.5)**2) * 20)
    # boundary points (hard-coded)
    tb = [tc, tc, tc, tc]
    xb = [jnp.array([[-1.]]), jnp.array([[1.]]), xc, xc]
    yb = [yc, yc, jnp.array([[-1.]]), jnp.array([[1.]])]
    return tc, xc, yc, ti, xi, yi, ui, tb, xb, yb


def generate_train_data(args, key, result_dir=None):
    eqn = args.equation
    if args.model == 'pinn':
        if eqn == 'diffusion3d':
            data = _pinn_train_generator_diffusion3d(
                args.nc, key
            )
        else:
            raise NotImplementedError
    elif args.model == 'spinn':
        if eqn == 'diffusion3d':
            data = _spinn_train_generator_diffusion3d(
                args.nc, key
            )
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return data


#============================== test dataset ===============================#
#------------------------- diffusion equation 3-d --------------------------#
@partial(jax.jit, static_argnums=(0, 1,))
def _test_generator_diffusion3d(model, data_dir):
    u_gt, tt = [], 0.
    for _ in range(101):
        u_gt += [jnp.load(os.path.join(data_dir, f'heat_gaussian_{tt:.2f}.npy'))]
        tt += 0.01
    u_gt = jnp.stack(u_gt)
    t = jnp.linspace(0., 1., u_gt.shape[0])
    x = jnp.linspace(-1., 1., u_gt.shape[1])
    y = jnp.linspace(-1., 1., u_gt.shape[2])
    t = jax.lax.stop_gradient(t)
    x = jax.lax.stop_gradient(x)
    y = jax.lax.stop_gradient(y)
    tm, xm, ym = jnp.meshgrid(t, x, y, indexing='ij')
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
    if eqn == 'diffusion3d':
        data = _test_generator_diffusion3d(
            args.model, args.data_dir
        )
    # else:
    #     raise NotImplementedError
    return data