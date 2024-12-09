import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import pdb
import matplotlib
import numpy as np

# 2d time-dependent klein-gordon exact u
def klein_gordon3d_exact_u(t, x, y, k):
    return (x + y) * jnp.cos(k * t) + (x * y) * jnp.sin(k * t)

def _klein_gordon3d(args, apply_fn, params, result_dir, e, resol=50):
    print("visualizing solution...")
    t = jnp.linspace(0., 10., resol)
    x = jnp.linspace(-1., 1., resol)
    y = jnp.linspace(-1., 1., resol)
    tm, xm, ym = jnp.meshgrid(t, x, y, indexing='ij')
    if args.model == 'pinn':
        t = tm.reshape(-1, 1)
        x = xm.reshape(-1, 1)
        y = ym.reshape(-1, 1)
    else:
        t = t.reshape(-1, 1)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

    u_ref = klein_gordon3d_exact_u(tm, xm, ym, args.k)

    os.makedirs(os.path.join(result_dir, f'vis/{e:05d}'), exist_ok=True)
    u_pred = apply_fn(params, t, x, y)
    if args.model == 'pinn':
        u_pred = u_pred.reshape(resol, resol, resol)
        u_ref = u_ref.reshape(resol, resol, resol)

    fig = plt.figure(figsize=(14, 5))

    # reference solution
    ax1 = fig.add_subplot(131, projection='3d')
    im = ax1.scatter(tm, xm, ym, c=u_ref, cmap = 'seismic', s=0.5)
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')
    ax1.set_zlabel('y')
    ax1.set_title(f'Reference $u(t, x, y)$', fontsize=15)

    # predicted solution
    ax2 = fig.add_subplot(132, projection='3d')
    im = ax2.scatter(tm, xm, ym, c=u_pred, cmap = 'seismic', s=0.5, vmin=jnp.min(u_ref), vmax=jnp.max(u_ref))
    ax2.set_xlabel('t')
    ax2.set_ylabel('x')
    ax2.set_zlabel('y')
    ax2.set_title(f'Predicted $u(t, x, y)$', fontsize=15)

    # absolute error
    ax3 = fig.add_subplot(133, projection='3d')
    im = ax3.scatter(tm, xm, ym, c=jnp.abs(u_ref-u_pred), cmap = 'seismic', s=0.5, vmin=jnp.min(u_ref), vmax=jnp.max(u_ref))
    ax3.set_xlabel('t')
    ax3.set_ylabel('x')
    ax3.set_zlabel('y')
    ax3.set_title(f'Absolute error', fontsize=15)

    cbar_ax = fig.add_axes([0.95, 0.3, 0.01, 0.4])
    cbar = fig.colorbar(im, cax=cbar_ax)

    plt.savefig(os.path.join(result_dir, f'vis/{e:05d}/pred.png'))
    plt.close()
    

def show_solution(args, apply_fn, params, test_data, result_dir, e, resol=50):
    if args.equation == 'klein_gordon3d':
        _klein_gordon3d(args, apply_fn, params, result_dir, e, resol)
    