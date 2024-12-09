from functools import partial

import jax
import jax.numpy as jnp


# 3d time-dependent flow-mixing exact solution
def flow_mixing3d_exact_u(t, x, y, omega):
    return -jnp.tanh((y/2)*jnp.cos(omega*t) - (x/2)*jnp.sin(omega*t))


# 3d time-dependent flow-mixing parameters
def flow_mixing3d_params(t, x, y, v_max, require_ab=False):
    # t, x, y must be meshgrid
    r = jnp.sqrt(x**2 + y**2)
    v_t = ((1/jnp.cosh(r))**2) * jnp.tanh(r)
    omega = (1/r)*(v_t/v_max)
    a, b = None, None
    if require_ab:
        a = -(v_t/v_max)*(y/r)
        b = (v_t/v_max)*(x/r)
    return omega, a, b