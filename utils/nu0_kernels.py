import jax
import jax.numpy as jnp

from functools import partial

from .special_functions import spherical_harmonics, scaled_spherical_bessel_i
from .dataset_processing import get_cartesian_vectors


@partial(jax.jit, static_argnames="all_species")
def compute_wk_nu0(jax_structures1, jax_structures2, all_species):

    wk_nu0_ii = {}
    s1 = jax_structures1["structure_indices"]
    s2 = jax_structures2["structure_indices"]
    where_ai1 = jax_structures1["atomic_indices_per_element"]
    where_ai2 = jax_structures2["atomic_indices_per_element"]
    s1_out = {}
    s2_out = {}

    for a_i in all_species:

        ai1_indices = where_ai1[a_i]
        ai2_indices = where_ai2[a_i]
        n1 = ai1_indices.shape[0]
        n2 = ai2_indices.shape[0]
        s1_ai = s1[ai1_indices]
        s2_ai = s2[ai2_indices]

        wk_nu0_ii[a_i] = jnp.ones((n1, n2))

        # Generate metadata for the nu=0 atom-wise kernels
        s1_out[a_i] = s1_ai
        s2_out[a_i] = s2_ai

    return wk_nu0_ii, s1_out, s2_out
