import jax
import jax.numpy as jnp


def get_rmse(first, second):
    return jnp.sqrt(jnp.mean((second-first)**2))
    

def get_mae(first, second):
    return jnp.mean(jnp.abs(second-first))
