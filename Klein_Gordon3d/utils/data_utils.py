from functools import partial

import jax
import jax.numpy as jnp

# 2d time-dependent klein-gordon exact u
def klein_gordon3d_exact_u(t, x, y, k):
    return (x + y) * jnp.cos(k * t) + (x * y) * jnp.sin(k * t)


# 2d time-dependent klein-gordon source term
def klein_gordon3d_source_term(t, x, y, k):
    u = klein_gordon3d_exact_u(t, x, y, k)
    return u**2 - (k**2)*u
