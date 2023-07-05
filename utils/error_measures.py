import jax
import jax.numpy as jnp


@jax.jit
def get_rmse(first, second):
    return jnp.sqrt(jnp.mean((second-first)**2))
    

@jax.jit
def get_mae(first, second):
    return jnp.mean(jnp.abs(second-first))
