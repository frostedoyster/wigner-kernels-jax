import jax
import jax.numpy as jnp

from functools import partial

from .special_functions import spherical_harmonics, scaled_spherical_bessel_i
from .dataset_processing import get_cartesian_vectors


def compute_wk_nu0(jax_structures1, jax_structures2, all_species):

    wk_nu0_ii = {}
    ai1 = jnp.concatenate(jax_structures1.atomic_species)
    ai2 = jnp.concatenate(jax_structures2.atomic_species)
    s1 = jax_structures1.structure_indices
    s2 = jax_structures2.structure_indices
    s1_out = {}
    s2_out = {}

    for a_i in all_species:

        ai1_indices = jnp.nonzero(ai1==a_i)[0]
        ai2_indices = jnp.nonzero(ai2==a_i)[0]
        n1 = ai1_indices.shape[0]
        n2 = ai2_indices.shape[0]
        s1_ai = s1[ai1_indices]
        s2_ai = s2[ai2_indices]

        wk_nu0_ii[a_i] = jnp.ones((n1, n2))

        # Generate metadata for the nu=0 atom-wise kernels
        s1_out[a_i] = s1_ai
        s2_out[a_i] = s2_ai

    return wk_nu0_ii, s1_out, s2_out
