
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch

from matplotlib.transforms import Affine2D
from matplotlib.patches import Ellipse
from matplotlib.colors import Normalize

rc('text', usetex=False)

def Helmholtz_2d_plot(it, y, x, u, u_gt, num_test, output_path, tag):
    # ship back to cpu
    y = y.detach().cpu().numpy().reshape(num_test, num_test)
    x = x.detach().cpu().numpy().reshape(num_test, num_test)
    u = u.detach().cpu().numpy().reshape(num_test, num_test)
    u_gt = u_gt.cpu().numpy().reshape(num_test, num_test)

    # plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].set_aspect('equal')
    col0 = axes[0].pcolormesh(x, y, u_gt, cmap='rainbow', shading='auto')
    axes[0].set_xlabel('x', fontsize=12, labelpad=12)
    axes[0].set_ylabel('y', fontsize=12, labelpad=12)
    axes[0].set_title('Exact U', fontsize=18, pad=18)
    div0 = make_axes_locatable(axes[0])
    cax0 = div0.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(col0, cax=cax0)

    axes[1].set_aspect('equal')
    col1 = axes[1].pcolormesh(x, y, u, cmap='rainbow', shading='auto')
    axes[1].set_xlabel('x', fontsize=12, labelpad=12)
    axes[1].set_ylabel('y', fontsize=12, labelpad=12)
    axes[1].set_title('Predicted U', fontsize=18, pad=18)
    div1 = make_axes_locatable(axes[1])
    cax1 = div1.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(col1, cax=cax1)

    axes[2].set_aspect('equal')
    col2 = axes[2].pcolormesh(x, y, np.abs(u-u_gt), cmap='rainbow', shading='auto')
    axes[2].set_xlabel('x', fontsize=12, labelpad=12)
    axes[2].set_ylabel('y', fontsize=12, labelpad=12)
    axes[2].set_title('Absolute error', fontsize=18, pad=18)
    div2 = make_axes_locatable(axes[2])
    cax2 = div2.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(col2, cax=cax2)
    cbar.mappable.set_clim(0, 1)

    
    plt.tight_layout()
    if it % 25 ==0:
        fig.savefig(output_path + "/{}_{}.png".format(tag, it))
    plt.clf()
    plt.close(fig)