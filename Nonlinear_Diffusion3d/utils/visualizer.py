import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import pdb
import matplotlib
import numpy as np

def _diffusion3d(args, apply_fn, params, test_data, result_dir, e, resol):
    print("visualizing solution...")

    nt = 11 # number of time steps to visualize
    t = jnp.linspace(0., 1., nt)
    x = jnp.linspace(-1., 1., resol)
    y = jnp.linspace(-1., 1., resol)
    xd, yd = jnp.meshgrid(x, y, indexing='ij')  # for 3-d surface plot
    tm, xm, ym = jnp.meshgrid(t, x, y, indexing='ij')
    if args.model == 'pinn':
        t = tm.reshape(-1, 1)
        x = xm.reshape(-1, 1)
        y = ym.reshape(-1, 1)
    else:
        t = t.reshape(-1, 1)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

    u_ref = test_data[-1]
    ref_idx = 0

    os.makedirs(os.path.join(result_dir, f'vis/{e:05d}'), exist_ok=True)
    u = apply_fn(params, t, x, y)
    if args.model == 'pinn':
        u = u.reshape(nt, resol, resol)
        u_ref = u_ref.reshape(-1, resol, resol)

    for tt in range(nt):
        fig = plt.figure(figsize=(12, 6))

        # reference solution (hard-coded; must be modified if nt changes)
        ax1 = fig.add_subplot(121, projection='3d')
        im = ax1.plot_surface(xd, yd, u_ref[ref_idx], cmap='jet', linewidth=0, antialiased=False)
        ref_idx += 10
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('u')
        ax1.set_title(f'Reference $u(x, y)$ at $t={tt*(1/(nt-1)):.1f}$', fontsize=15)
        ax1.set_zlim(jnp.min(u_ref), jnp.max(u_ref))

        # predicted solution
        ax2 = fig.add_subplot(122, projection='3d')
        im = ax2.plot_surface(xd, yd, u[tt], cmap='jet', linewidth=0, antialiased=False)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('u')
        ax2.set_title(f'Predicted $u(x, y)$ at $t={tt*(1/(nt-1)):.1f}$', fontsize=15)
        ax2.set_zlim(jnp.min(u_ref), jnp.max(u_ref))

        plt.savefig(os.path.join(result_dir, f'vis/{e:05d}/pred_{tt*(1/(nt-1)):.1f}.png'))
        plt.close()



def show_solution(args, apply_fn, params, test_data, result_dir, e, resol=50):
    _diffusion3d(args, apply_fn, params, test_data, result_dir, e, resol)
    