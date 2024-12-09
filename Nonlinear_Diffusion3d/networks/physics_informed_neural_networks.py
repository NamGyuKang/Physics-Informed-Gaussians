import pdb
from typing import Sequence

import jax.numpy as jnp
from flax import linen as nn
from functools import partial
from typing import Any, Callable, Sequence, Tuple, Optional, Union, Dict

from flax import linen as nn
from flax.core.frozen_dict import freeze

from jax import random, jit, vmap
import jax.numpy as jnp
from jax.nn.initializers import glorot_normal, normal, zeros, constant, uniform
import numpy as np
from jax import lax


activation_fn = {
    "relu": nn.relu,
    "gelu": nn.gelu,
    "swish": nn.swish,
    "sigmoid": nn.sigmoid,
    "tanh": nn.activation.tanh,
    "sin": jnp.sin,
}


def _get_activation(str):
    if str in activation_fn:
        return activation_fn[str]

    else:
        raise NotImplementedError(f"Activation {str} not supported yet!")

class Gaussian3d_Diag(nn.Module):
    num_gaussian: int = 100
    grid_range: float = 2.
    sigmas_range : float = 0.5
    mlp_dim: int= 4
    # step_idx:int = 1

    def setup(self):
        self.mu_x = self.param("mu_x", uniform(self.grid_range), (self.mlp_dim, self.num_gaussian,))
        self.mu_y = self.param("mu_y", uniform(self.grid_range), (self.mlp_dim, self.num_gaussian,))
        self.mu_z = self.param("mu_z", uniform(self.grid_range), (self.mlp_dim, self.num_gaussian,))
        self.sigmas = self.param("sigmas", constant(self.sigmas_range), (self.mlp_dim, self.num_gaussian, 3))
        self.weight = self.param("weight", normal(), (self.mlp_dim, self.num_gaussian, 1))

    @nn.compact
    def __call__(self, x, y, z):
        sigmas_x = self.sigmas[:,:,0]
        sigmas_y = self.sigmas[:,:,1]
        sigmas_z = self.sigmas[:,:,2]
        
        # diffusion 3d
        y = ((y+1.) / 2.) * self.grid_range
        z = ((z+1.) / 2.) * self.grid_range


        pdf = 0.5 * (((x[None,:,:] - self.mu_x[:,None,:])/sigmas_x[:,None,:])**2 + ((y[None,:,:] - self.mu_y[:,None,:])/sigmas_y[:,None,:])**2 + ((z[None,:,:] - self.mu_z[:,None,:])/sigmas_z[:,None,:])**2)
        pdf = jnp.exp(-pdf)

        rasterized_color_primes = pdf * self.weight.squeeze()[:, None, :]

        output = rasterized_color_primes.sum(2)
        
        return output.T


class Gaussian3d_Full(nn.Module):
    num_gaussian: int
    grid_range: float
    sigmas_range : float
    mlp_dim : int

    def setup(self):
        self.mu_x = self.param("mu_x", uniform(self.grid_range), (self.mlp_dim, self.num_gaussian))
        self.mu_y = self.param("mu_y", uniform(self.grid_range), (self.mlp_dim, self.num_gaussian))
        self.mu_z = self.param("mu_z", uniform(self.grid_range), (self.mlp_dim, self.num_gaussian))
        self.sigmas = self.param("sigmas", constant(self.sigmas_range), (self.mlp_dim, self.num_gaussian, 3))
        self.r1 = self.param("r1", constant(1.0), (self.mlp_dim, 1, self.num_gaussian, 1))
        self.r2 = self.param("r2", constant(0.0), (self.mlp_dim, 1, self.num_gaussian, 1))
        self.r3 = self.param("r3", constant(0.0), (self.mlp_dim, 1, self.num_gaussian, 1))
        self.r4 = self.param("r4", constant(0.0), (self.mlp_dim, 1, self.num_gaussian, 1))
        self.weight = self.param("weight", normal(), (self.mlp_dim, self.num_gaussian, 1))
        
        self.sigmas = self.sigmas[:,None,:,:]
        self.mu_x = self.mu_x[:, None, :, None]
        self.mu_y = self.mu_y[:, None, :, None]
        self.mu_z = self.mu_z[:, None, :, None]

        r = jnp.concatenate([self.r1, self.r2, self.r3, self.r4], -1)
        L = self.build_scaling_rotation(self.sigmas, r)
        self.cov = L @ L.transpose(0, 1, 2, 4, 3)

    @nn.compact
    def __call__(self, x, y, z):
        # diffusion 3d
        y = ((y+1.) / 2.) * self.grid_range
        z = ((z+1.) / 2.) * self.grid_range


        mu = jnp.concatenate([self.mu_x, self.mu_y, self.mu_z], -1)
        X = jnp.concatenate([x[None, :, None, :], y[None, :, None, :], z[None, :, None, :] ], -1)
        d = X - mu

        out = self.cov @ d[..., None]
        out = d[..., None, :] @ out
        out = out.squeeze()

        pdf = jnp.exp(-0.5*out)

        rasterized_color_primes = pdf * self.weight.squeeze()[:, None, :]
        output = rasterized_color_primes.sum(2)

        return output.T 

    def build_rotation(self, r):
        norm = jnp.sqrt(r[..., 0]*r[..., 0] + r[..., 1]*r[..., 1] + r[..., 2]*r[..., 2] + r[..., 3]*r[..., 3])

        q = r / norm[..., None]

        R = jnp.zeros((q.shape[0], q.shape[1], q.shape[2], 3, 3))

        r = q[..., 0]
        x = q[..., 1]
        y = q[..., 2]
        z = q[..., 3]

        R = R.at[..., 0, 0].set(1 - 2 * (y*y + z*z))
        R = R.at[..., 0, 1].set(2 * (x*y - r*z))
        R = R.at[..., 0, 2].set(2 * (x*z + r*y))
        R = R.at[..., 1, 0].set(2 * (x*y + r*z))
        R = R.at[..., 1, 1].set(1 - 2 * (x*x + z*z))
        R = R.at[..., 1, 2].set(2 * (y*z - r*x))
        R = R.at[..., 2, 0].set(2 * (x*z - r*y))
        R = R.at[..., 2, 1].set(2 * (y*z + r*x))
        R = R.at[..., 2, 2].set(1 - 2 * (x*x + y*y))
        return R

    def build_scaling_rotation(self, s, r):
        L = jnp.zeros((s.shape[0],s.shape[1],s.shape[2], 3, 3))
        R = self.build_rotation(r)
        
        L = L.at[..., 0, 0].set(1./s[...,0])
        L = L.at[..., 1, 1].set(1./s[...,1])
        L = L.at[..., 2, 2].set(1./s[...,2])
        L = R @ L
        return L


class PINN3d(nn.Module):
    features: Sequence[int]
    out_dim: int
    pos_enc: int
    num_gaussian: int = 100
    grid_range: float = 2.
    sigmas_range : float = 15.
    mlp_dim: int= 4

    def setup(self):
        self.activation_fn = _get_activation('tanh')

    @nn.compact
    def __call__(self, x, y, z):
        X = Gaussian3d_Diag(self.num_gaussian, self.grid_range, self.sigmas_range, self.mlp_dim)(x,y,z)
        # X = Gaussian3d_Full(self.num_gaussian, self.grid_range, self.sigmas_range, self.mlp_dim)(x,y,z)
        
        init = nn.initializers.glorot_normal()
        for fs in self.features[:-1]:
            X = nn.Dense(fs, kernel_init=init)(X)
            X = self.activation_fn(X)
        X = nn.Dense(self.features[-1], kernel_init=init)(X)

        return X


class SPINN3d(nn.Module):
    features: Sequence[int]
    r: int
    out_dim: int
    pos_enc: int
    mlp: str

    @nn.compact
    def __call__(self, x, y, z):
        '''
        inputs: input factorized coordinates
        outputs: feature output of each body network
        xy: intermediate tensor for feature merge btw. x and y axis
        pred: final model prediction (e.g. for 2d output, pred=[u, v])
        '''
        if self.pos_enc != 0:
            # positional encoding only to spatial coordinates
            freq = jnp.expand_dims(jnp.arange(1, self.pos_enc+1, 1), 0)
            y = jnp.concatenate((jnp.ones((y.shape[0], 1)), jnp.sin(y@freq), jnp.cos(y@freq)), 1)
            z = jnp.concatenate((jnp.ones((z.shape[0], 1)), jnp.sin(z@freq), jnp.cos(z@freq)), 1)

            # causal PINN version (also on time axis)
            #  freq_x = jnp.expand_dims(jnp.power(10.0, jnp.arange(0, 3)), 0)
            # x = x@freq_x
            
        inputs, outputs, xy, pred = [x, y, z], [], [], []
        init = nn.initializers.glorot_normal()

        if self.mlp == 'mlp':
            for X in inputs:
                for fs in self.features[:-1]:
                    X = nn.Dense(fs, kernel_init=init)(X)
                    X = nn.activation.tanh(X)
                X = nn.Dense(self.r*self.out_dim, kernel_init=init)(X)
                outputs += [jnp.transpose(X, (1, 0))]

        elif self.mlp == 'modified_mlp':
            for X in inputs:
                U = nn.activation.tanh(nn.Dense(self.features[0], kernel_init=init)(X))
                V = nn.activation.tanh(nn.Dense(self.features[0], kernel_init=init)(X))
                H = nn.activation.tanh(nn.Dense(self.features[0], kernel_init=init)(X))
                for fs in self.features[:-1]:
                    Z = nn.Dense(fs, kernel_init=init)(H)
                    Z = nn.activation.tanh(Z)
                    H = (jnp.ones_like(Z)-Z)*U + Z*V
                H = nn.Dense(self.r*self.out_dim, kernel_init=init)(H)
                outputs += [jnp.transpose(H, (1, 0))]
        
        for i in range(self.out_dim):
            xy += [jnp.einsum('fx, fy->fxy', outputs[0][self.r*i:self.r*(i+1)], outputs[1][self.r*i:self.r*(i+1)])]
            pred += [jnp.einsum('fxy, fz->xyz', xy[i], outputs[-1][self.r*i:self.r*(i+1)])]

        if len(pred) == 1:
            # 1-dimensional output
            return pred[0]
        else:
            # n-dimensional output
            return pred


