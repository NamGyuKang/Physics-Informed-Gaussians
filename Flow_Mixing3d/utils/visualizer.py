import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import pdb
import matplotlib
import numpy as np

def _test_generator_flow_mixing3d(nc_test, v_max):
    t = jnp.linspace(0, 4, nc_test)
    x = jnp.linspace(-4, 4, nc_test)
    y = jnp.linspace(-4, 4, nc_test)
    tm, xm, ym = jnp.meshgrid(t, x, y, indexing='ij')

    omega, _, _ = flow_mixing3d_params(tm, xm, ym, v_max)
    u_gt = flow_mixing3d_exact_u(tm, xm, ym, omega)

    t = tm.reshape(-1, 1)
    x = xm.reshape(-1, 1)
    y = ym.reshape(-1, 1)
    u_gt = u_gt.reshape(-1, 1)

    return t, x, y, u_gt

def flow_mixing3d_exact_u(t, x, y, omega):
    return -jnp.tanh((y/2)*jnp.cos(omega*t) - (x/2)*jnp.sin(omega*t))

def flow_mixing3d_params(t, x, y, v_max, require_ab = False):
    r = jnp.sqrt(x**2 + y**2)
    v_t = ((1/jnp.cosh(r))**2) * jnp.tanh(r)
    omega = (1/r)*(v_t/v_max)
    a, b = None, None

    if require_ab:
        a = -(v_t/ v_max)*(y/r)
        b = (v_t/v_max)*(x/r)
    return omega, a, b 
    
def _flow_mixing3d(args, apply_fn, params, result_dir, e, resol=50):
    print("visualizing solution...")
    t = jnp.linspace(0., 4., resol)
    x = jnp.linspace(-4., 4., resol)
    y = jnp.linspace(-4., 4., resol)
    tm, xm, ym = jnp.meshgrid(t, x, y, indexing='ij')
    t = tm.reshape(-1, 1)
    x = xm.reshape(-1, 1)
    y = ym.reshape(-1, 1)

    os.makedirs(os.path.join(result_dir, f'vis/{e:05d}'), exist_ok=True)
    u_pred = apply_fn(params, t, x, y)
    u_pred = u_pred.reshape(resol, resol, resol)
    u_ref = _test_generator_flow_mixing3d(resol, 0.385)[3].reshape(resol, resol, resol)
    
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(2, 3)
    plt.subplot(gs[0, 0])
    # reference solution
    plt.pcolormesh(u_ref[0, :, :].T,cmap = 'jet')
    plt.title(f'Reference $u(t, x, y)$, t=0', fontsize=25)

    plt.subplot(gs[0, 1])
    plt.pcolormesh(u_ref[25, :, :].T,cmap = 'jet')
    plt.title(f'Reference $u(t, x, y)$, t=2', fontsize=25)

    plt.subplot(gs[0, 2])
    plt.pcolormesh(u_ref[-1, :, :].T,cmap = 'jet')
    plt.title(f'Reference $u(t, x, y)$, t=4', fontsize=25)
    
    # predicted solution
    plt.subplot(gs[1, 0])
    plt.pcolormesh(u_pred[0, :, :].T,cmap = 'jet')
    plt.title(f'Predicted $u(t, x, y)$, t=0', fontsize=25)

    plt.subplot(gs[1, 1])
    plt.pcolormesh(u_pred[25, :, :].T,cmap = 'jet')
    plt.title(f'Predicted $u(t, x, y)$, t=2', fontsize=25)

    plt.subplot(gs[1, 2])
    plt.pcolormesh(u_pred[-1, :, :].T,cmap = 'jet')
    plt.title(f'Predicted $u(t, x, y)$, t=4', fontsize=25)


    # # fig = plt.figure(figsize=(14, 5))

    # # reference solution
    # plt.subplot(gs[2, 0])
    # ax1 = fig.add_subplot(131, projection='3d')
    # im = ax1.scatter(tm, xm, ym, c=u_ref, cmap = 'seismic', s=0.5)
    # ax1.set_xlabel('x')
    # ax1.set_ylabel('y')
    # ax1.set_zlabel('z')
    # ax1.set_title(f'Reference $u(x, y, z)$', fontsize=15)

    # # predicted solution
    # plt.subplot(gs[2, 1])
    # ax2 = fig.add_subplot(132, projection='3d')
    # im = ax2.scatter(tm, xm, ym, c=u_pred, cmap = 'seismic', s=0.5, vmin=jnp.min(u_ref), vmax=jnp.max(u_ref))
    # ax2.set_xlabel('x')
    # ax2.set_ylabel('y')
    # ax2.set_zlabel('z')
    # ax2.set_title(f'Predicted $u(t, x, y)$', fontsize=15)

    # # absolute error
    # plt.subplot(gs[2, 2])
    # ax3 = fig.add_subplot(133, projection='3d')
    # im = ax3.scatter(tm, xm, ym, c=jnp.abs(u_ref-u_pred), cmap = 'seismic', s=0.5, vmin=jnp.min(u_ref), vmax=jnp.max(u_ref))
    # ax3.set_xlabel('x')
    # ax3.set_ylabel('y')
    # ax3.set_zlabel('z')
    # ax3.set_title(f'Absolute error', fontsize=15)

    # cbar_ax = fig.add_axes([0.95, 0.3, 0.01, 0.4])
    # fig.colorbar(im, cax=cbar_ax)

    plt.savefig(os.path.join(result_dir, f'vis/{e:05d}/pred.png'))
    plt.close()

def show_solution(args, apply_fn, params, test_data, result_dir, e, resol=50):
    _flow_mixing3d(args, apply_fn, params, result_dir, e, resol)