import os
import pdb
from functools import partial

import jax
import jax.numpy as jnp
import optax
import scipy.io
from networks.physics_informed_neural_networks import *

import ml_collections

def setup_networks(args, key):
    # build network
    dim = args.equation[-2:]
    if args.model == 'pinn':
        # feature sizes
        feat_sizes = tuple([args.features for _ in range(args.n_layers - 1)] + [args.out_dim])
        if dim == '3d':
            model = PINN3d(feat_sizes, args.out_dim, args.pos_enc, args.num_gaussian, args.grid_range, args.sigmas_range, args.mlp_dim)
        else:
            raise NotImplementedError
    else: # SPINN
        # feature sizes
        if dim == '3d':
            model = SPINN3d(feat_sizes, args.r, args.out_dim, args.pos_enc, args.mlp)
        else:
            raise NotImplementedError
    # initialize params
    # dummy inputs must be given
    if dim == '3d':
        params = model.init(
                key,
                jnp.ones((args.nc, 1)),
                jnp.ones((args.nc, 1)),
                jnp.ones((args.nc, 1))
            )
    else:
        raise NotImplementedError

    return jax.jit(model.apply), params


def name_model(args):
    name = [
        f'ng{args.num_gaussian}',
        f'gr{args.grid_range}',
        f'sr{args.sigmas_range}',
        f'k{args.mlp_dim}',
        f'nl{args.n_layers}',
        f'fs{args.features}',
        f'lr{args.lr}',
        f's{args.seed}',
        f'r{args.r}'
    ]
    if args.model != 'spinn':
        del name[-1]
    name.insert(0, f'nc{args.nc}')
    if args.equation == 'klein_gordon3d':
        name.append(f'k{args.k}')
    name.append(f'{args.mlp}')
        
    return '_'.join(name)


def save_config(args, result_dir):
    with open(os.path.join(result_dir, 'configs.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f'{arg}: {getattr(args, arg)}\n')


# single update function
@partial(jax.jit, static_argnums=(0,))
def update_model(optim, gradient, params, state):
    updates, state = optim.update(gradient, state)
    params = optax.apply_updates(params, updates)
    return params, state

